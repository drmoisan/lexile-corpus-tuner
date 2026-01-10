"""
CLI entry point and orchestration for atomic executor.

Provides argument parsing, workspace validation, and main execution loop
that coordinates PlanParser, FeatureResolver, QCRunner, and PromptBuilder.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import os
import selectors
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import IO, cast

from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver
from scripts.dev_tools.atomic_executor.plan_discovery import resolve_feature_plan
from scripts.dev_tools.atomic_executor.plan_parser import PlanParser, PlanTask
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_runner import QCRunner

DEFAULT_PROMPT_TEMPLATE = ".github/prompts/execute-plan-template.md"
PROTECTED_BRANCHES = {"main", "master", "development"}
LOG_DIR = ".agent_logs"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
        argv (list[str]): Command-line arguments (typically sys.argv[1:]).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Atomic task-by-task executor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("path", help="Feature folder path OR a plan.md path.")
        sp.add_argument(
            "--workspace",
            default=None,
            help="Repo root (defaults to auto-detect).",
        )
        sp.add_argument(
            "--feature",
            default=None,
            help="Feature folder name under docs/features/active (optional).",
        )
        sp.add_argument(
            "--prompt-template",
            default=DEFAULT_PROMPT_TEMPLATE,
            help="Prompt template path.",
        )
        sp.add_argument(
            "--start",
            default=None,
            help="Start at a specific task id like P2-T3.",
        )
        sp.add_argument(
            "--max-fix-attempts",
            type=int,
            default=2,
            help="Retries for current task if QC fails.",
        )
        sp.add_argument(
            "--print-prompt",
            action="store_true",
            help="Print resolved prompt for current task and exit.",
        )
        sp.add_argument(
            "--copy-prompt",
            action="store_true",
            help="Copy resolved prompt to clipboard (and exit).",
        )
        sp.add_argument(
            "--preferred-model",
            default=None,
            help=(
                "Preferred AI model (Copilot CLI --model value or display name), "
                "e.g. 'gpt-5.1-codex-max' or 'Claude Sonnet 4.5'."
            ),
        )

    sp_exec = sub.add_parser("execute", help="Execute from first unchecked or --start.")
    add_common(sp_exec)

    sp_resume = sub.add_parser("resume", help="Resume from first unchecked task.")
    add_common(sp_resume)

    sp_all = sub.add_parser("execute-all", help="Execute all remaining tasks.")
    add_common(sp_all)

    return p.parse_args(argv)


def resolve_workspace(workspace_arg: str | None) -> Path:
    """
    Resolve workspace root directory.

    Args:
        workspace_arg (str | None): Explicit workspace path from CLI.

    Returns:
        Path: Resolved workspace root.
    """
    if workspace_arg:
        return Path(workspace_arg).resolve()

    # Infer: assume this file lives at <repo>/scripts/dev_tools/atomic_executor/
    return Path(__file__).resolve().parents[3]


def ensure_clean_tree(workspace: Path) -> None:
    """
    Verify working tree is clean (no uncommitted changes).

    Args:
        workspace (Path): Repository root.

    Raises:
        RuntimeError: If working tree has uncommitted changes.
        FileNotFoundError: If git executable not found.
    """
    # Use shutil.which for cross-platform git resolution (avoids hardcoding
    # /usr/bin/git or C:\Program Files\Git\bin\git.exe).
    git_exe = shutil.which("git")
    if not git_exe:
        raise FileNotFoundError("Required executable not found on PATH: git")

    result = (
        subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [git_exe, "status", "--porcelain"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
    )
    if result.stdout.strip():
        raise RuntimeError("Working tree is not clean. Commit/stash before running.")


def refuse_protected_branch(workspace: Path) -> None:
    """
    Refuse execution on protected branches.

    Args:
        workspace (Path): Repository root.

    Raises:
        RuntimeError: If current branch is protected.
    """
    branch = _current_branch(workspace)
    if branch and branch in PROTECTED_BRANCHES:
        raise RuntimeError(f"Refusing to run on protected branch '{branch}'.")


def _current_branch(workspace: Path) -> str | None:
    """Get current git branch name, or None if error."""
    # Use shutil.which for cross-platform git resolution.
    git_exe = shutil.which("git")
    if not git_exe:
        return None

    try:
        result = subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [git_exe, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        b = result.stdout.strip()
        return b or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_clipboard_command() -> list[str] | None:
    """
    Detect the correct clipboard command for the current platform.

    Purpose:
        Platform-aware clipboard command detection with WSL support.

    Returns:
        list[str] | None: Command and arguments if available,
            None if no clipboard support.

    Side Effects:
        None - pure detection function.
    """
    # Detect platform
    if sys.platform == "win32":
        candidates: list[list[str]] = [["clip"]]
    elif sys.platform == "darwin":
        candidates = [["pbcopy"]]
    else:  # Linux/Unix
        # Check for WSL (reports linux but needs Windows clipboard)
        is_wsl = False
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    is_wsl = True
        except FileNotFoundError:
            pass

        if is_wsl:
            candidates = [
                ["clip.exe"],  # WSL prefers Windows clipboard
                ["pbcopy"],  # Fallback if macOS tools installed
                ["wl-copy"],  # Wayland
                ["xclip", "-selection", "clipboard"],  # X11
                ["xsel", "--clipboard", "--input"],  # X11 alternative
            ]
        else:
            candidates = [
                ["wl-copy"],  # Wayland
                ["xclip", "-selection", "clipboard"],  # X11
                ["xsel", "--clipboard", "--input"],  # X11 alternative
            ]

    # Validate candidates exist on PATH
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd

    return None


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard using platform-appropriate command.

    Purpose:
        Provides clipboard access via explicit platform detection + validation.

    Args:
        text (str): Text to copy.

    Returns:
        bool: True if successful,
            False if no clipboard command available or copy failed.

    Side Effects:
        Executes system clipboard command (clip/pbcopy/xclip/etc.).
    """

    def _try_pyperclip_copy() -> bool:
        """
        Attempt copy via optional pyperclip dependency.

        Returns:
            bool: True when pyperclip is available and succeeds, otherwise False
            to allow fallback to platform-specific commands.
        """
        try:
            import pyperclip  # type: ignore[import-untyped]
        except ImportError:
            return False

        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    if _try_pyperclip_copy():
        return True

    # Get platform-appropriate clipboard command
    cmd = get_clipboard_command()
    if not cmd:
        return False

    # Execute clipboard command with validation
    exe = shutil.which(cmd[0])
    if not exe:
        return False

    try:
        subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [exe, *cmd[1:]],
            input=text,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_copilot(
    *,
    workspace: Path,
    prompt_text: str,
    log_file: Path,
    task_id: str,
    preferred_model: str | None,
    run_id: str,
    resume_session: bool = False,
    _idle_timeout_seconds: float | None = None,
) -> None:
    """
    Invoke GitHub Copilot CLI with prompt and tool permissions.

    Args:
        workspace (Path): Repository root.
        prompt_text (str): Complete prompt to execute.
        log_file (Path): Path to log file for output.
        task_id (str): Current task id (used for log labeling).
        preferred_model (str | None): Preferred model name or Copilot CLI --model value.
        run_id (str): Run id for grouping per-task artifacts.
        resume_session (bool): Reuse prior Copilot session for this task if True.

    Raises:
        FileNotFoundError: If the `copilot` CLI executable is not available.
        CalledProcessError: If Copilot CLI execution fails.

    Side Effects:
        - Executes `copilot` CLI command
        - Writes to log file
        - Writes a per-task session share markdown file
    """

    def normalize_copilot_model(model: str) -> str:
        """
        Normalize a human-facing model name into a Copilot CLI --model choice.

        Purpose:
            Users may provide either the slash-command display name (e.g.
            "GPT-5.1-Codex-Max") or the CLI choice key
            (e.g. "gpt-5.1-codex-max"). This normalizes to a valid --model value.

        Args:
            model (str): User-provided model string.

        Returns:
            str: Normalized Copilot CLI model identifier.

        Raises:
            ValueError: If the model cannot be normalized.
        """
        raw = model.strip()
        if not raw:
            raise ValueError("Model name cannot be empty")

        # Known Copilot CLI v0.0.375 model choice identifiers.
        # Keep this small, explicit, and aligned to `copilot help` output.
        known_choices = {
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "claude-opus-4.5",
            "claude-sonnet-4",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5.1-codex-mini",
            "gpt-5-mini",
            "gpt-4.1",
            "gemini-3-pro-preview",
        }

        # Fast-path: already a valid choice.
        lowered = raw.lower()
        if lowered in known_choices:
            return lowered

        # Display-name normalization: strip parentheses, collapse whitespace,
        # replace spaces with hyphens, and standardize common punctuation.
        cleaned = lowered
        cleaned = cleaned.replace("(preview)", "preview")
        cleaned = cleaned.replace("(", " ").replace(")", " ")
        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.replace(" ", "-")
        cleaned = cleaned.replace("--", "-")

        if cleaned in known_choices:
            return cleaned

        raise ValueError(f"Unsupported Copilot CLI model: {model}")

    def is_vscode_copilot_shim(exe_path: str) -> bool:
        """
        Identify the VS Code Copilot Chat extension shim.

        Purpose:
            The VS Code extension may create `copilot.ps1`/`copilot.bat` shims that
            prompt to install the real Copilot CLI. The atomic executor needs the
            real agentic CLI (installed via WinGet/Homebrew/npm), not an
            interactive installer shim.

        Args:
            exe_path (str): Resolved executable path from `shutil.which()`.

        Returns:
            bool: True if the path looks like the VS Code shim, otherwise False.
        """
        norm = exe_path.replace("/", "\\").lower()

        # Normalize repeated backslashes (helps in tests and when paths are
        # string-escaped by tooling).
        while "\\\\" in norm:
            norm = norm.replace("\\\\", "\\")
        return "\\code\\user\\globalstorage\\github.copilot-chat\\copilotcli\\" in norm

    # Find copilot on PATH, skipping VS Code shims.
    # shutil.which() only returns the first match, but the VS Code extension
    # shim may appear first. Search all PATH entries for a non-shim copilot.
    copilot_exe = None
    path_env = os.environ.get("PATH", "")
    for path_dir in path_env.split(os.pathsep):
        for candidate_name in ["copilot.exe", "copilot.bat", "copilot"]:
            candidate = Path(path_dir) / candidate_name
            if candidate.exists() and not is_vscode_copilot_shim(str(candidate)):
                copilot_exe = str(candidate)
                break
        if copilot_exe:
            break

    if not copilot_exe:
        raise FileNotFoundError(
            "Required executable not found on PATH: copilot. "
            "Install GitHub Copilot CLI via either: "
            "winget install GitHub.Copilot  OR  npm install -g @github/copilot"
        )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    share_dir = log_file.parent / "copilot_sessions"
    share_dir.mkdir(parents=True, exist_ok=True)
    share_path = share_dir / f"copilot_session_{run_id}_{task_id}.md"
    if resume_session and not share_path.exists():
        share_path.touch()

    # Write prompt to temporary file to avoid Windows command-line length limits
    # (WinError 206: filename or extension too long when prompt passed via -p).
    prompt_dir = log_file.parent / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"prompt_{run_id}_{task_id}.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    argv: list[str] = [
        copilot_exe,
    ]

    normalized_model: str | None = None
    if preferred_model:
        normalized_model = normalize_copilot_model(preferred_model)
        argv.extend(["--model", normalized_model])

    supports_sessions = _copilot_supports_session(copilot_exe)
    if resume_session and supports_sessions:
        argv.extend(["--session-path", str(share_path)])
    elif resume_session and not supports_sessions:
        _log_msg(
            log_file,
            "INFO: Copilot CLI does not support --session-path; skipping resume.",
        )

    argv.extend(
        [
            "--share",
            str(share_path),
            "--allow-tool",
            "write",
            "--allow-tool",
            "shell(poetry)",
            "--allow-tool",
            "shell(python)",
            "--allow-tool",
            "shell(git)",
        ]
    )

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n\n=== Copilot invocation ===\n")
        f.write(f"task_id: {task_id}\n")
        if preferred_model:
            f.write(f"preferred_model: {preferred_model}\n")
        if normalized_model:
            f.write(f"normalized_model: {normalized_model}\n")
        if resume_session:
            f.write(f"resume_session: {supports_sessions}\n")
        f.write(f"share_path: {share_path}\n")
        f.write(f"prompt_file: {prompt_file}\n")
        f.write("(prompt omitted from log for brevity; use --print-prompt to view)\n")
        f.flush()

        # Pass prompt via stdin to avoid Windows command-line length limits.
        # Use Popen to stream stdout to both console and log file.
        # Use binary mode + incremental decoding to avoid Python
        # TextIOWrapper buffering.
        with prompt_file.open("rb") as prompt_f:
            process = subprocess.Popen(  # noqa: S603 - static analysis can't verify runtime validation
                argv,
                cwd=workspace,
                stdin=prompt_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            idle_timeout = _resolve_idle_timeout_seconds(_idle_timeout_seconds)
            _stream_copilot_output(
                process=process,
                decoder=decoder,
                log_file=f,
                task_id=task_id,
                idle_timeout_seconds=idle_timeout,
            )

    # Post-processing: deduplicate prompt from session file
    _clean_session_file(share_path, prompt_text)


def _resolve_idle_timeout_seconds(configured: float | None) -> float | None:
    """Resolve the idle timeout value from argument or environment.

    An idle timeout of ``None`` disables hang detection. A value ``<= 0`` also
    disables the timeout. Environment variable
    ``ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS`` overrides the default when
    the helper is invoked without an explicit timeout.
    """

    if configured is not None:
        return configured if configured > 0 else None

    env_val = os.environ.get("ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS")
    if env_val is None:
        return 300.0

    env_val = env_val.strip()
    if not env_val:
        return 300.0

    try:
        parsed = float(env_val)
    except ValueError:
        return 300.0

    return parsed if parsed > 0 else None


def _stream_copilot_output(
    *,
    process: subprocess.Popen[bytes],
    decoder: codecs.IncrementalDecoder,
    log_file: IO[str],
    task_id: str,
    idle_timeout_seconds: float | None,
) -> None:
    """Stream Copilot output with hang detection.

    Purpose:
        Avoid silent hangs when the Copilot CLI is waiting for interactive
        input by enforcing an idle timeout on stdout activity. Terminates the
        process and raises ``TimeoutError`` when exceeded.
    """

    # Selector-based non-blocking read to avoid hard blocking on stdout.
    selector = selectors.DefaultSelector()
    if process.stdout:
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
        except ValueError:
            # Some test doubles may not expose a real file descriptor; skip
            # registration and rely on poll/idle timeout instead.
            pass

    last_activity = time.monotonic()

    while True:
        # Poll stdout readiness with a small timeout to keep the loop
        # responsive while still allowing idle timeout checks.
        events = selector.select(timeout=0.1)

        # Drain any ready streams.
        for key, _ in events:
            stream_obj = key.fileobj
            stream = cast(IO[bytes], stream_obj)
            read1 = getattr(stream, "read1", None)
            chunk: bytes
            if callable(read1):
                chunk = cast(bytes, read1(4096))
            else:
                chunk = stream.read(4096)

            if chunk:
                text_chunk = decoder.decode(chunk, final=False)
                if text_chunk:
                    print(text_chunk, end="", flush=True)
                    log_file.write(text_chunk)
                    log_file.flush()
                last_activity = time.monotonic()
            else:
                selector.unregister(stream_obj)

        # Break once the process finishes and all streams are drained.
        if process.poll() is not None and not selector.get_map():
            break

        # Hang detection based on idle time (no output + still running).
        if idle_timeout_seconds is not None:
            idle_duration = time.monotonic() - last_activity
            if idle_duration > idle_timeout_seconds:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
                raise TimeoutError(
                    "Copilot CLI produced no output for "
                    f"{idle_timeout_seconds} seconds while executing task "
                    f"{task_id}; terminated to avoid hanging."
                )

    # Flush decoder tail.
    remaining = decoder.decode(b"", final=True)
    if remaining:
        print(remaining, end="", flush=True)
        log_file.write(remaining)
        log_file.flush()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


@lru_cache(maxsize=4)
def _copilot_supports_session(copilot_exe: str) -> bool:
    """
    Detect whether the copilot CLI supports session reuse flags.

    Returns:
        bool: True if --session-path is supported.
    """
    try:
        result = subprocess.run(  # noqa: S603 - copilot_exe resolved via shutil.which
            [copilot_exe, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    return "--session-path" in result.stdout


def _clean_session_file(session_path: Path, prompt_text: str) -> None:
    """
    Remove the prompt from the beginning of the session file to avoid duplication.

    Args:
        session_path (Path): Path to the generated session markdown file.
        prompt_text (str): The prompt text that was sent to the agent.
    """
    if not session_path.exists():
        return

    try:
        content = session_path.read_text(encoding="utf-8")
        # Check whether the file begins with the prompt text.
        # Allow small implementation differences in the echoed header.
        if content.startswith(prompt_text):
            # Slice it off
            cleaned_content = content[len(prompt_text) :].lstrip()
            # If nothing remains, arguably we should leave it empty or keep something?
            # Usually there is subsequent conversation.
            # Add a header to indicate this is the transcript
            cleaned_content = "# Copilot Session Transcript\n\n" + cleaned_content
            session_path.write_text(cleaned_content, encoding="utf-8")
    except Exception as e:
        # Don't fail the build if cosmetic cleanup fails
        print(
            f"Warning: Failed to clean session file {session_path}: {e}",
            file=sys.stderr,
        )


def _log_msg(log_file: Path, msg: str) -> None:
    """Write message to log file and flush."""
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def _execute_one_task(
    workspace: Path,
    cur: PlanTask,
    parser: PlanParser,
    builder: PromptBuilder,
    qc_runner: QCRunner,
    log_file: Path,
    prompt_template_path: Path,
    max_fix_attempts: int,
    feature_dir: Path,
    preferred_model: str | None,
    run_id: str,
    print_prompt: bool = False,
    copy_prompt: bool = False,
) -> int:
    """
    Execute a single atomic task with retries.

    Args:
        workspace (Path): Repo root.
        cur (PlanTask): The task to execute.
        parser (PlanParser): Plan parser instance (for updates).
        builder (PromptBuilder): Prompt builder instance.
        qc_runner (QCRunner): QC runner instance.
        log_file (Path): Path to log file.
        prompt_template_path (Path): Path to prompt template.
        max_fix_attempts (int): Max number of retries (0 = infinite).
        feature_dir (Path): Active feature directory.
        preferred_model (str | None): Preferred AI model name to force in Copilot CLI.
        run_id (str): Run id for grouping per-task artifacts.
        print_prompt (bool): If True, print prompt and return.
        copy_prompt (bool): If True, copy prompt and return.

    Returns:
        int: Exit code (0 = success, 5 = failed).
    """
    # Handle --print-prompt / --copy-prompt (static preview)
    if print_prompt or copy_prompt:
        # Initial build without retry context for preview
        prompt_text = builder.build(feature_dir, cur)
        if print_prompt:
            print(prompt_text)
            return 0

        if copy_prompt:
            ok = copy_to_clipboard(prompt_text)
            if not ok:
                print(
                    "Clipboard copy not available; prompt printed below.",
                    file=sys.stderr,
                )
                print(prompt_text)
            else:
                print(
                    f"Prompt copied to clipboard for task {cur.task_id}.",
                    file=sys.stderr,
                )
            return 0

    attempt = 1
    retry_ctx = None

    while True:
        if max_fix_attempts > 0 and attempt > max_fix_attempts:
            msg = (
                f"Failed to complete task {cur.task_id} after "
                f"{max_fix_attempts} attempts."
            )
            print(msg, file=sys.stderr)
            _log_msg(log_file, f"ERROR: {msg}")
            print(f"See log: {log_file}", file=sys.stderr)
            return 5

        # Rebuild prompt with retry context if applicable
        prompt_text = builder.build(feature_dir, cur, retry_context=retry_ctx)

        limit_str = str(max_fix_attempts) if max_fix_attempts > 0 else "∞"
        msg = f"Executing task {cur.task_id} (attempt {attempt}/{limit_str})"
        print(msg)
        _log_msg(log_file, f"INFO: {msg}")

        run_copilot(
            workspace=workspace,
            prompt_text=prompt_text,
            log_file=log_file,
            task_id=cur.task_id,
            preferred_model=preferred_model,
            run_id=run_id,
            resume_session=attempt > 1,
        )

        # Refresh plan/task state after Copilot run
        cur_after = parser.find_task_by_id(cur.task_id)

        # Task-step QC (scoped)
        try:
            qc_runner.run_scoped()
        except subprocess.CalledProcessError as e:
            err_msg = f"Scoped QC failed for task {cur.task_id}: {e}"
            print(err_msg, file=sys.stderr)
            _log_msg(log_file, f"WARN: {err_msg}")

            # Prepare context for next attempt
            retry_ctx = (
                f"Attempt {attempt} failed verification.\n"
                f"Error: {e}\n"
                "Please fix code/test issues and try again."
            )
            attempt += 1
            continue

        # Flip checkbox if model didn't do it (authoritative edit after QC)
        if cur_after and not cur_after.checked:
            parser.flip_checkbox(cur_after)

        success_msg = f"Task {cur.task_id} complete and gated."
        print(success_msg)
        _log_msg(log_file, f"SUCCESS: {success_msg}")
        return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for atomic executor CLI.

    Purpose:
        Orchestrates feature folder resolution, plan parsing, QC execution,
        and Copilot invocation for one task at a time.

    Args:
        argv (list[str]): Command-line arguments.

    Returns:
        int: Exit code (0 for success, non-zero for error).

    Side Effects:
        - Validates workspace state (git clean, not on protected branch)
        - Parses and modifies plan.md
        - Runs QC toolchains
        - Invokes Copilot CLI
        - Writes log files
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    workspace = resolve_workspace(args.workspace)

    # Preconditions: not on protected branch
    # ensure_clean_tree(workspace) - Disabled to allow mid-execution restarts
    refuse_protected_branch(workspace)

    # Resolve feature folder
    active_dir = workspace / "docs" / "features" / "active"
    resolver = FeatureResolver(workspace, active_dir)
    _, feature_dir = resolver.resolve(args.path, args.feature)

    try:
        resolved_plan = resolve_feature_plan(feature_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    plan_path = resolved_plan.path
    prompt_template_path = (workspace / args.prompt_template).resolve()

    if not prompt_template_path.is_file():
        print(
            f"Prompt template not found: {prompt_template_path}",
            file=sys.stderr,
        )
        return 2

    # Setup logging
    log_dir = workspace / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    import datetime

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = log_dir / f"atomic_executor_{run_id}.log"

    # Parse plan and preflight validate
    parser = PlanParser(plan_path)
    parser.preflight_validate()

    # Determine current task
    if args.cmd == "resume":
        cur = parser.next_unchecked_task()
        if cur is None:
            print("Plan already complete: no unchecked tasks found.")
            return 0
    elif args.cmd == "execute-all":
        cur = parser.next_unchecked_task()
        if cur is None:
            print("Plan already complete: no unchecked tasks found.")
            return 0
    else:
        if args.start:
            cur = parser.find_task_by_id(args.start)
        else:
            cur = parser.next_unchecked_task()
            if cur is None:
                print("Plan already complete: no unchecked tasks found.")
                return 0

    builder = PromptBuilder(
        workspace,
        prompt_template_path,
        preferred_model=args.preferred_model,
    )
    qc_runner = QCRunner(workspace)

    while True:
        # Build prompt and execute
        result = _execute_one_task(
            workspace=workspace,
            cur=cur,
            parser=parser,
            builder=builder,
            qc_runner=qc_runner,
            log_file=log_file,
            prompt_template_path=prompt_template_path,
            max_fix_attempts=args.max_fix_attempts,
            feature_dir=feature_dir,
            preferred_model=args.preferred_model,
            run_id=run_id,
            print_prompt=args.print_prompt,
            copy_prompt=args.copy_prompt,
        )

        if result != 0:
            return result

        # Stop here if interactive command (print/copy)
        if args.print_prompt or args.copy_prompt:
            return 0

        # Check phase completion after task success
        if parser.phase_complete(cur.phase):
            print(f"Phase {cur.phase} complete -> running full toolchain...")
            try:
                qc_runner.run_full()
            except subprocess.CalledProcessError as e:
                print(
                    f"Full QC failed after completing Phase {cur.phase}: {e}",
                    file=sys.stderr,
                )
                return 5

        # If not execute-all, we are done after one task
        if args.cmd != "execute-all":
            print("Next: run 'resume' for the next task.")
            return 0

        # If execute-all, find next task
        next_task = parser.next_unchecked_task()
        if next_task is None:
            print("All tasks complete.")
            return 0
        cur = next_task
        print(f"Proceeding to next task: {cur.task_id}...")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

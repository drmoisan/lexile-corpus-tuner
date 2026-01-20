"""
CLI entry point and orchestration for atomic executor.

Provides argument parsing, workspace validation, and main execution loop
that coordinates PlanParser, FeatureResolver, QCRunner, and PromptBuilder.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, cast

from scripts.dev_tools.atomic_executor.copilot_runner import CopilotRunResult
from scripts.dev_tools.atomic_executor.copilot_throttling import (
    CallRateLimiter,
    ExponentialBackoff,
    FailureKind,
    SystemClock,
    SystemRandom,
    TimeSleeper,
    classify_copilot_failure,
)
from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver
from scripts.dev_tools.atomic_executor.plan_discovery import resolve_feature_plan
from scripts.dev_tools.atomic_executor.plan_parser import (
    AutoQCPhase,
    PlanParser,
    PlanTask,
)
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_runner import QCLoopResult, QCRunner

DEFAULT_PROMPT_TEMPLATE = ".github/prompts/execute-plan-template.md"
PROTECTED_BRANCHES = {"main", "master", "development"}
LOG_DIR = ".agent_logs"
EXECUTOR_LOCK_FILE = ".agent_logs/executor.lock"
EXECUTOR_LOCK_BYPASS_ENV = "ATOMIC_EXECUTOR_SKIP_LOCK"

# Safe, bounded defaults for Copilot CLI throttling controls (issue #80).
DEFAULT_COPILOT_CLI_MAX_CALLS_PER_WINDOW = 6
DEFAULT_COPILOT_CLI_WINDOW_SECONDS = 60.0
DEFAULT_COPILOT_CLI_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_COPILOT_CLI_BACKOFF_MAX_SECONDS = 60.0
DEFAULT_COPILOT_CLI_OUTPUT_TAIL_BYTES = 4096
DEFAULT_COPILOT_CLI_MAX_RETRIES = 8
DEFAULT_COPILOT_AGENT = "atomic_execution"
DEFAULT_COPILOT_ALLOW_SHELL = True
DEFAULT_COPILOT_ALLOW_ALL_PATHS = True
DEFAULT_COPILOT_ALLOW_ALL_URLS = False
DEFAULT_COPILOT_TRUST_WORKSPACE = True

# When Copilot CLI cannot request approval (common in headless/non-interactive
# runs), it emits this exact substring and may then stall until an idle-timeout.
# We detect it during output streaming and fail fast with actionable guidance.
COPILOT_PERMISSION_DENIED_SUBSTRING = (
    "Permission denied and could not request permission from user"
)


class CopilotPermissionDeniedError(RuntimeError):
    """Raised when Copilot output indicates an approval/permission dead-end.

    Purpose:
        Provides a typed signal that the Copilot CLI emitted the known
        permission-denied substring and is unlikely to recover without
        additional permissions or an interactive approval path.
    """


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

        sp.add_argument(
            "--copilot-cli-max-calls-per-window",
            type=int,
            default=DEFAULT_COPILOT_CLI_MAX_CALLS_PER_WINDOW,
            help=(
                "Max Copilot CLI calls per time window "
                "(call-rate based; not token based)."
            ),
        )
        sp.add_argument(
            "--copilot-cli-window-seconds",
            type=float,
            default=DEFAULT_COPILOT_CLI_WINDOW_SECONDS,
            help="Window size in seconds for call-rate limiting.",
        )
        sp.add_argument(
            "--copilot-cli-backoff-base-seconds",
            type=float,
            default=DEFAULT_COPILOT_CLI_BACKOFF_BASE_SECONDS,
            help="Base seconds for exponential backoff after throttling.",
        )
        sp.add_argument(
            "--copilot-cli-backoff-max-seconds",
            type=float,
            default=DEFAULT_COPILOT_CLI_BACKOFF_MAX_SECONDS,
            help="Maximum seconds for exponential backoff cap after throttling.",
        )
        sp.add_argument(
            "--copilot-cli-output-tail-bytes",
            type=int,
            default=DEFAULT_COPILOT_CLI_OUTPUT_TAIL_BYTES,
            help=(
                "Number of Copilot output bytes to retain as an in-memory tail for "
                "throttling classification and error messages."
            ),
        )
        sp.add_argument(
            "--copilot-cli-max-retries",
            type=int,
            default=DEFAULT_COPILOT_CLI_MAX_RETRIES,
            help="Max throttle-triggered retries per atomic task (bounded by default).",
        )

        sp.add_argument(
            "--copilot-allow-shell",
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_COPILOT_ALLOW_SHELL,
            help=(
                "Allow all shell commands without approval (adds --allow-tool shell)."
            ),
        )
        sp.add_argument(
            "--copilot-allow-all-paths",
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_COPILOT_ALLOW_ALL_PATHS,
            help=("Allow Copilot CLI to access any path without per-path approvals."),
        )
        sp.add_argument(
            "--copilot-allow-all-urls",
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_COPILOT_ALLOW_ALL_URLS,
            help=("Allow Copilot CLI to access any URL without per-URL approvals."),
        )
        sp.add_argument(
            "--copilot-trust-workspace",
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_COPILOT_TRUST_WORKSPACE,
            help=("Ensure the workspace is listed in Copilot CLI trusted_folders."),
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


def acquire_executor_lock(workspace: Path) -> Path:
    """
    Acquire the single-run lock to prevent concurrent executor sessions.

    Purpose:
        Ensures only one execute-all run is active at a time so that
        `--continue` does not resume unrelated Copilot sessions.
        Allows pytest subprocesses launched by the executor to bypass the
        lock when ATOMIC_EXECUTOR_SKIP_LOCK is set or when pytest exports
        PYTEST_CURRENT_TEST.

    Args:
        workspace (Path): Repository root used to resolve the lock file path.

    Returns:
        Path: The resolved lock file path.

    Raises:
        RuntimeError: If the lock file already exists.
    """
    lock_path = workspace / EXECUTOR_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Allow pytest subprocesses spawned by the executor to bypass the lock so
    # execute-all tests do not fail when the parent process holds the lock.
    if os.getenv(EXECUTOR_LOCK_BYPASS_ENV) == "1" or os.getenv("PYTEST_CURRENT_TEST"):
        return lock_path

    if lock_path.exists():
        raise RuntimeError(
            f"Atomic executor lock already exists: {lock_path.as_posix()}"
        )

    lock_path.write_text("atomic_executor_lock\n", encoding="utf-8")
    return lock_path


def release_executor_lock(lock_path: Path) -> None:
    """
    Release the single-run lock file if it exists.

    Args:
        lock_path (Path): Path to the lock file to remove.
    """
    with contextlib.suppress(FileNotFoundError):
        if lock_path.exists():
            lock_path.unlink()


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
    is_first_task: bool = True,
    allow_all_paths: bool = DEFAULT_COPILOT_ALLOW_ALL_PATHS,
    allow_all_urls: bool = DEFAULT_COPILOT_ALLOW_ALL_URLS,
    allow_shell: bool = DEFAULT_COPILOT_ALLOW_SHELL,
    trust_workspace: bool = DEFAULT_COPILOT_TRUST_WORKSPACE,
    _idle_timeout_seconds: float | None = None,
    _output_tail_bytes: int | None = None,
) -> CopilotRunResult:
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
        is_first_task (bool): True for the first task in a plan run.
        allow_all_paths (bool): Allow all path access without per-path approvals.
        allow_all_urls (bool): Allow all URL access without per-URL approvals.
        allow_shell (bool): Allow all shell commands without per-command approvals.
        trust_workspace (bool): Persist workspace path in Copilot trusted_folders.

    Returns:
        CopilotRunResult: Exit code and a bounded output tail snippet.

    Raises:
        FileNotFoundError: If the `copilot` CLI executable is not available.
        TimeoutError: If Copilot produces no output for the idle timeout.

    Side Effects:
        - Executes `copilot` CLI command
        - Writes to log file
        - Writes a per-task session share markdown file
    """

    output_tail_bytes = 4096 if _output_tail_bytes is None else _output_tail_bytes
    if output_tail_bytes < 0:
        output_tail_bytes = 0

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
        # Normalize separators so we can detect shim paths across:
        # - Windows local VS Code (Code/User/globalStorage/...)
        # - VS Code Remote / devcontainers (.vscode-server/.../globalStorage/...)
        norm = exe_path.replace("\\", "/").lower()

        # Normalize repeated slashes (helps in tests and when paths are
        # string-escaped by tooling).
        while "//" in norm:
            norm = norm.replace("//", "/")

        # The key reliable signature is the Copilot Chat extension storage path.
        return "/github.copilot-chat/" in norm and "/copilotcli/" in norm

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

    # Trust the workspace up front when requested so headless runs avoid the
    # interactive trust prompt.
    # Gate workspace trust update explicitly so callers can opt out.
    if trust_workspace:
        _ensure_trusted_workspace(workspace=workspace)

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
        "--agent",
        DEFAULT_COPILOT_AGENT,
    ]

    normalized_model: str | None = None
    if preferred_model:
        normalized_model = normalize_copilot_model(preferred_model)
        argv.extend(["--model", normalized_model])

    # Session continuation: --continue is the default for all tasks after the first.
    # --session-path is used only for explicit resume (e.g., after executor restart).
    use_continue = False
    if resume_session:
        argv.extend(["--session-path", str(share_path)])
    elif not is_first_task:
        argv.append("--continue")
        use_continue = True

    argv.extend(
        [
            "--share",
            str(share_path),
            "--add-dir",
            str(workspace),
            "--allow-tool",
            "write",
            "--allow-tool",
            "shell(poetry)",
            "--allow-tool",
            "shell(python)",
            "--allow-tool",
            "shell(python3)",
            "--allow-tool",
            "shell(git)",
        ]
    )

    # Expand permissions for headless sessions when explicitly enabled.
    if allow_shell:
        argv.extend(["--allow-tool", "shell"])
    if allow_all_paths:
        argv.append("--allow-all-paths")
    if allow_all_urls:
        argv.append("--allow-all-urls")

    argv.extend(
        [
            "-p",
            f"Follow these instructions exactly: @{prompt_file}",
        ]
    )

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n\n=== Copilot invocation ===\n")
        f.write(f"task_id: {task_id}\n")
        if preferred_model:
            f.write(f"preferred_model: {preferred_model}\n")
        if normalized_model:
            f.write(f"normalized_model: {normalized_model}\n")
        session_mode = (
            "continue" if use_continue else "resume" if resume_session else "new"
        )
        f.write(f"session_mode: {session_mode}\n")
        f.write(f"share_path: {share_path}\n")
        f.write(f"prompt_file: {prompt_file}\n")
        f.write("(prompt omitted from log for brevity; use --print-prompt to view)\n")
        f.flush()

        # Use Popen to stream stdout to both console and log file.
        # Use binary mode + incremental decoding to avoid Python
        # TextIOWrapper buffering.
        process = subprocess.Popen(  # noqa: S603 - static analysis can't verify runtime validation
            argv,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        idle_timeout = _resolve_idle_timeout_seconds(_idle_timeout_seconds)
        try:
            exit_code, output_tail = _stream_copilot_output(
                process=process,
                decoder=decoder,
                log_file=f,
                task_id=task_id,
                idle_timeout_seconds=idle_timeout,
                output_tail_bytes=output_tail_bytes,
            )
        except CopilotPermissionDeniedError as exc:
            # Fail fast with actionable context instead of waiting for the idle-timeout.
            argv_summary = " ".join(argv)
            raise RuntimeError(
                "Copilot CLI reported a permissions dead-end and cannot request "
                "approval from the user in this environment. "
                f"Detected: {COPILOT_PERMISSION_DENIED_SUBSTRING!r}. "
                f"argv: {argv_summary}. "
                "Guidance: ensure the executor uses programmatic mode (-p/--prompt) "
                "and includes explicit tool and directory permissions "
                "(e.g. --allow-tool write, --allow-tool shell(poetry), "
                "--allow-tool shell(python3), --allow-tool shell(git), "
                "--allow-tool shell, --allow-all-paths, "
                "and --add-dir <workspace>). "
                "If policy blocks headless execution, run the command interactively to "
                "grant approvals."
            ) from exc

    # Post-processing: deduplicate prompt from session file
    _clean_session_file(share_path, prompt_text)

    return CopilotRunResult(exit_code=exit_code, output_tail=output_tail)


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


def _copilot_config_dir() -> Path:
    """Resolve the Copilot CLI configuration directory.

    Purpose:
        Locate the Copilot CLI configuration directory using XDG conventions
        when available, and the default ~/.copilot path otherwise.

    Returns:
        Path: Directory that should contain Copilot CLI config.json.

    Raises:
        None.

    Side Effects:
        None.
    """
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    # Prefer XDG config location when explicitly configured.
    if xdg_home and xdg_home.strip():
        return Path(xdg_home).expanduser().resolve() / "copilot"

    return Path.home() / ".copilot"


def _ensure_trusted_workspace(*, workspace: Path) -> None:
    """Ensure the workspace appears in Copilot CLI trusted_folders.

    Purpose:
        Headless runs cannot accept the interactive trust prompt. This helper
        records the workspace path in Copilot's config so programmatic mode can
        access files without blocking.

    Args:
        workspace (Path): Repository root to trust.

    Returns:
        None.

    Raises:
        RuntimeError: When the Copilot CLI config.json is malformed.

    Side Effects:
        Creates or updates ~/.copilot/config.json (or XDG_CONFIG_HOME) to
        include the workspace path in trusted_folders.
    """
    config_dir = _copilot_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    config_data: dict[str, object] = {}
    # Load the existing config if present; otherwise start from defaults.
    if config_file.exists():
        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Copilot CLI config.json is invalid JSON. "
                f"Fix or remove: {config_file}"
            ) from exc

    trusted_folders = config_data.get("trusted_folders")
    # Normalize trusted_folders into a list for safe updates.
    if trusted_folders is None:
        trusted_folders_list: list[str] = []
    elif isinstance(trusted_folders, list):
        # Normalize trusted folder entries to strings for stable comparisons.
        trusted_folders_list = [
            str(item) for item in cast(list[object], trusted_folders)
        ]
    else:
        raise RuntimeError(
            "Copilot CLI config.json has non-list trusted_folders. "
            f"Fix: {config_file}"
        )

    workspace_path = str(workspace.resolve())
    # Only append when the workspace is not already trusted.
    if workspace_path not in trusted_folders_list:
        trusted_folders_list.append(workspace_path)
        config_data["trusted_folders"] = trusted_folders_list
        config_file.write_text(
            json.dumps(config_data, indent=2, sort_keys=True), encoding="utf-8"
        )


def _stream_copilot_output(
    *,
    process: subprocess.Popen[bytes],
    decoder: codecs.IncrementalDecoder,
    log_file: IO[str],
    task_id: str,
    idle_timeout_seconds: float | None,
    output_tail_bytes: int | None,
) -> tuple[int, str]:
    """Stream Copilot output with hang detection.

    Purpose:
        Avoid silent hangs when the Copilot CLI is waiting for interactive
        input by enforcing an idle timeout on stdout activity. Terminates the
        process and raises ``TimeoutError`` when exceeded.

        While streaming, retain a bounded tail buffer of the raw output bytes.
        This tail is returned to callers for throttling classification and
        actionable error messages.
    """

    def _terminate_process(process_to_kill: subprocess.Popen[bytes]) -> None:
        """Attempt to terminate the Copilot process without assuming APIs exist.

        Purpose:
            In production, ``subprocess.Popen`` provides ``kill()`` and
            ``terminate()`` methods. In unit tests, we often stub ``Popen`` with
            a minimal mock object. This helper makes termination best-effort so
            tests can focus on behavior rather than strict process mechanics.

        Args:
            process_to_kill (subprocess.Popen[bytes]): The process to stop.

        Returns:
            None

        Side Effects:
            Attempts to stop the process and waits briefly for it to exit.
        """
        kill_fn = getattr(process_to_kill, "kill", None)
        term_fn = getattr(process_to_kill, "terminate", None)

        # Prefer kill() (hard stop), then terminate() (soft stop).
        if callable(kill_fn):
            kill_fn()
        elif callable(term_fn):
            term_fn()

        with contextlib.suppress(subprocess.TimeoutExpired, AttributeError):
            process_to_kill.wait(timeout=5)

    # Retain a bounded output tail in bytes so throttling classification can be
    # performed without reading the log file or depending on exception
    # stdout/stderr.
    output_tail_bytes = 0 if output_tail_bytes is None else output_tail_bytes
    if output_tail_bytes < 0:
        output_tail_bytes = 0
    tail_buffer = bytearray()

    # Cross-platform streaming approach:
    # - `selectors` / `select.select` cannot monitor pipes on Windows, and will
    #   raise WinError 10038/10022. A background reader thread avoids that.
    # - The main thread maintains idle-timeout enforcement.
    q: queue.Queue[bytes | None] = queue.Queue()

    def _reader() -> None:
        """Read bytes from stdout until EOF and push them to the queue."""

        stream = process.stdout
        if stream is None:
            q.put(None)
            return

        read1 = getattr(stream, "read1", None)
        try:
            # Continuously drain Copilot stdout in chunks so the main thread can
            # enforce idle timeouts without blocking on reads.
            while True:
                chunk: bytes
                if callable(read1):
                    chunk = cast(bytes, read1(4096))
                else:
                    chunk = stream.read(4096)

                if not chunk:
                    break

                q.put(chunk)
        finally:
            q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    last_activity = time.monotonic()
    saw_eof = False

    # Track a small rolling decoded window to detect known fail-fast substrings
    # even when the bytes arrive split across chunks.
    permission_scan_window = ""
    permission_scan_window_max_chars = 2048

    # Consume output opportunistically while enforcing idle-timeout termination
    # if Copilot produces no output and remains running.
    while True:
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            item = None

        if item is None:
            # Distinguish between "no data right now" (queue.Empty) and EOF.
            if not reader_thread.is_alive() and not saw_eof:
                saw_eof = True
        else:
            if output_tail_bytes > 0:
                tail_buffer.extend(item)
                if len(tail_buffer) > output_tail_bytes:
                    del tail_buffer[:-output_tail_bytes]

            text_chunk = decoder.decode(item, final=False)
            if text_chunk:
                # Fail fast if Copilot cannot request permission from the user.
                permission_scan_window = (permission_scan_window + text_chunk)[
                    -permission_scan_window_max_chars:
                ]
                if COPILOT_PERMISSION_DENIED_SUBSTRING in permission_scan_window:
                    _terminate_process(process)
                    raise CopilotPermissionDeniedError(
                        COPILOT_PERMISSION_DENIED_SUBSTRING
                    )
                print(text_chunk, end="", flush=True)
                log_file.write(text_chunk)
                log_file.flush()
            last_activity = time.monotonic()

        # Break once the process finishes AND the reader has reached EOF.
        if process.poll() is not None and saw_eof and q.empty():
            break

        # Hang detection based on idle time (no output + still running).
        if idle_timeout_seconds is not None and process.poll() is None:
            idle_duration = time.monotonic() - last_activity
            if idle_duration > idle_timeout_seconds:
                _terminate_process(process)
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
    return (return_code, tail_buffer.decode("utf-8", errors="replace"))


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


READ_TASK_PATTERN = re.compile(r"^read\b", re.IGNORECASE)


def _is_phase0_read_task(task: PlanTask) -> bool:
    """
    Determine whether a task is a Phase 0 "read" task.

    Purpose:
        Allows the executor to bundle Phase 0 read tasks into the first prompt.

    Args:
        task (PlanTask): Task to evaluate.

    Returns:
        bool: True when the task is Phase 0 and begins with "read".
    """
    if task.phase != 0:
        return False
    return bool(READ_TASK_PATTERN.match(task.title.strip()))


def _phase0_read_tasks(parser: PlanParser) -> list[PlanTask]:
    """
    Collect unchecked Phase 0 read tasks in plan order.

    Purpose:
        Provides the executor and prompt builder a deterministic list of
        tasks to bundle into the first prompt.

    Args:
        parser (PlanParser): Parsed plan access.

    Returns:
        list[PlanTask]: Unchecked Phase 0 read tasks.
    """
    plan = parser.parse()
    read_tasks: list[PlanTask] = []

    # Preserve phase/task ordering for deterministic prompt sequencing.
    for task in sorted(plan.tasks, key=lambda x: (x.phase, x.task_num)):
        if task.checked:
            continue
        if _is_phase0_read_task(task):
            read_tasks.append(task)

    return read_tasks


def _first_non_read_task(parser: PlanParser) -> PlanTask | None:
    """
    Return the first unchecked task that is not a Phase 0 read task.

    Purpose:
        Ensures the first prompt can combine Phase 0 reads with the first
        actionable non-read task.

    Args:
        parser (PlanParser): Parsed plan access.

    Returns:
        PlanTask | None: First non-read task, or None if none exist.
    """
    plan = parser.parse()

    # Scan tasks in order while skipping Phase 0 read tasks.
    for task in sorted(plan.tasks, key=lambda x: (x.phase, x.task_num)):
        if task.checked:
            continue
        if _is_phase0_read_task(task):
            continue
        return task

    return None


def _build_qc_fix_prompt(
    *,
    feature_dir: Path,
    phase: AutoQCPhase,
    failure: QCLoopResult,
) -> str:
    """
    Build a focused prompt for fixing QC failures in an auto-QC phase.

    Purpose:
        Provide a concise, fix-only prompt that directs the LLM to resolve
        QC failures without running the toolchain itself.

    Args:
        feature_dir (Path): Feature directory for context.
        phase (AutoQCPhase): Auto-detected QC phase metadata.
        failure (QCLoopResult): Failure result with output details.

    Returns:
        str: Prompt text for Copilot CLI execution.
    """
    failure_detail = failure.failure
    # Guard against missing failure detail to keep prompt construction safe.
    if failure_detail is None:
        return "Auto-QC failure: unknown failure detail."

    # Build a readable list of artifact outputs for the fixer.
    artifact_lines: list[str] = []
    for step, path in phase.artifact_paths.items():
        artifact_lines.append(f"- {step}: {path.as_posix()}")
    artifact_list = "\n".join(artifact_lines)

    return (
        "You are fixing an auto-executed QC phase in the atomic executor.\n\n"
        "Context:\n"
        f"- Feature folder: {feature_dir.as_posix()}\n"
        f"- QC phase: {phase.phase}\n\n"
        "Failure:\n"
        f"- Step: {failure_detail.step}\n"
        f"- Exit code: {failure_detail.returncode}\n\n"
        "Captured output:\n"
        f"{failure_detail.output}\n\n"
        "Artifacts (already written):\n"
        f"{artifact_list}\n\n"
        "Instructions:\n"
        "- Fix the reported issues in the codebase.\n"
        "- Do NOT run the toolchain yourself; the executor will rerun it.\n"
        "- Keep changes minimal and scoped to the failure.\n"
        "- When done, reply with a brief summary of what you changed.\n"
    )


def _execute_auto_qc_phase(
    *,
    workspace: Path,
    phase: AutoQCPhase,
    parser: PlanParser,
    qc_runner: QCRunner,
    log_file: Path,
    feature_dir: Path,
    preferred_model: str | None,
    run_id: str,
    copilot_rate_limiter: CallRateLimiter,
    copilot_backoff: ExponentialBackoff,
    copilot_max_retries: int,
    copilot_output_tail_bytes: int,
    copilot_allow_shell: bool,
    copilot_allow_all_paths: bool,
    copilot_allow_all_urls: bool,
    copilot_trust_workspace: bool,
    max_fix_attempts: int,
    print_prompt: bool = False,
    copy_prompt: bool = False,
    is_first_task: bool = True,
) -> int:
    """
    Execute an auto-detected QC phase without per-task LLM calls.

    Purpose:
        Run the toolchain loop in Python, capture artifacts, and invoke Copilot
        only when a QC step fails.

    Returns:
        int: Exit code (0 = success, 5 = failure).
    """
    if print_prompt or copy_prompt:
        print(
            "Auto-QC phase detected; no prompt is generated for this task.",
            file=sys.stderr,
        )
        return 0

    attempt = 1

    # Retry the QC loop until it passes or the max attempt limit is reached.
    while True:
        if max_fix_attempts > 0 and attempt > max_fix_attempts:
            msg = (
                f"Failed to complete auto-QC phase {phase.phase} after "
                f"{max_fix_attempts} attempts."
            )
            print(msg, file=sys.stderr)
            _log_msg(log_file, f"ERROR: {msg}")
            print(f"See log: {log_file}", file=sys.stderr)
            return 5

        try:
            result = qc_runner.run_full_loop_with_artifacts(
                artifact_paths=phase.artifact_paths,
            )
        except RuntimeError as exc:
            err_msg = f"Auto-QC phase {phase.phase} failed: {exc}"
            print(err_msg, file=sys.stderr)
            _log_msg(log_file, f"ERROR: {err_msg}")
            return 5

        # Successful toolchain loop => mark the QC tasks complete.
        if result.success:
            # Mark all auto-QC tasks as complete after the loop passes.
            # Flip each QC task checkbox so the plan reflects completion.
            for task_id in phase.task_ids:
                current_task = parser.find_task_by_id(task_id)
                if not current_task.checked:
                    parser.flip_checkbox(current_task)

            success_msg = f"Auto-QC phase {phase.phase} complete and gated."
            print(success_msg)
            _log_msg(log_file, f"SUCCESS: {success_msg}")
            return 0

        # If the loop failed but we lack detail, stop with actionable logs.
        if result.failure is None:
            err_msg = f"Auto-QC phase {phase.phase} failed without error details."
            print(err_msg, file=sys.stderr)
            _log_msg(log_file, f"ERROR: {err_msg}")
            return 5

        # Build fix-only prompt and invoke Copilot when QC fails.
        prompt_text = _build_qc_fix_prompt(
            feature_dir=feature_dir,
            phase=phase,
            failure=result,
        )

        copilot_invocation = 0
        throttle_retries = 0

        # Retry Copilot invocation on throttling without rerunning the loop yet.
        while True:
            copilot_rate_limiter.acquire()

            copilot_result = run_copilot(
                workspace=workspace,
                prompt_text=prompt_text,
                log_file=log_file,
                task_id=f"AUTO-QC-P{phase.phase}",
                preferred_model=preferred_model,
                run_id=run_id,
                resume_session=(attempt > 1 or copilot_invocation > 0),
                is_first_task=is_first_task,
                allow_all_paths=copilot_allow_all_paths,
                allow_all_urls=copilot_allow_all_urls,
                allow_shell=copilot_allow_shell,
                trust_workspace=copilot_trust_workspace,
                _output_tail_bytes=copilot_output_tail_bytes,
            )
            copilot_invocation += 1

            # Successful Copilot run clears throttle backoff state.
            if copilot_result.exit_code == 0:
                copilot_backoff.on_success()
                break

            failure_kind = classify_copilot_failure(
                exit_code=copilot_result.exit_code,
                output_tail=copilot_result.output_tail,
            )
            # Fail fast on non-throttle Copilot failures.
            if failure_kind is FailureKind.NON_THROTTLE:
                err_msg = (
                    "Copilot CLI failed (non-throttle) while fixing auto-QC. "
                    f"exit_code={copilot_result.exit_code}. "
                    f"output_tail={copilot_result.output_tail!r}"
                )
                print(err_msg, file=sys.stderr)
                _log_msg(log_file, f"ERROR: {err_msg}")
                print(f"See log: {log_file}", file=sys.stderr)
                return 5

            # Normalize retry ceiling so throttling does not loop forever.
            if copilot_max_retries < 0:
                copilot_max_retries = 0

            # Stop once we exhaust throttle retries.
            if throttle_retries >= copilot_max_retries:
                err_msg = (
                    f"Copilot CLI appears throttled during auto-QC fixes, "
                    f"max retries ({copilot_max_retries}) exhausted."
                )
                print(err_msg, file=sys.stderr)
                _log_msg(log_file, f"ERROR: {err_msg}")
                print(f"See log: {log_file}", file=sys.stderr)
                return 5

            delay_seconds = copilot_backoff.on_throttle()
            throttle_retries += 1

            retry_msg = (
                "Copilot throttled during auto-QC fixes; retry "
                f"{throttle_retries}/{copilot_max_retries} after "
                f"{delay_seconds:.2f}s backoff."
            )
            print(retry_msg)
            _log_msg(log_file, f"WARN: {retry_msg}")

            # Apply backoff delay via injected sleeper for deterministic tests.
            if delay_seconds > 0:
                copilot_rate_limiter.sleeper.sleep(delay_seconds)

        attempt += 1


def execute_one_task(
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
    copilot_rate_limiter: CallRateLimiter,
    copilot_backoff: ExponentialBackoff,
    copilot_max_retries: int,
    copilot_output_tail_bytes: int,
    copilot_allow_shell: bool,
    copilot_allow_all_paths: bool,
    copilot_allow_all_urls: bool,
    copilot_trust_workspace: bool,
    include_phase0_reads: bool,
    print_prompt: bool = False,
    copy_prompt: bool = False,
    is_first_task: bool = True,
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
        preferred_model (str | None): Preferred AI model name to force in Copilot
            CLI.
        run_id (str): Run id for grouping per-task artifacts.
        copilot_rate_limiter (CallRateLimiter): Per-run call limiter that caps the
            number of Copilot CLI invocations per time window.
        copilot_backoff (ExponentialBackoff): Backoff strategy used when a Copilot
            call appears throttled.
        copilot_max_retries (int): Max number of throttle-triggered retries per
            atomic task.
        copilot_output_tail_bytes (int): Number of bytes of Copilot output tail to
            retain for failure classification.
        copilot_allow_shell (bool): Allow all shell commands without approval.
        copilot_allow_all_paths (bool): Allow all file paths without approval.
        copilot_allow_all_urls (bool): Allow all URLs without approval.
        copilot_trust_workspace (bool): Persist workspace in Copilot trusted list.
        include_phase0_reads (bool): Whether to include Phase 0 read tasks in
            the prompt before the current task.
        print_prompt (bool): If True, print prompt and return.
        copy_prompt (bool): If True, copy prompt and return.
        is_first_task (bool): True when this is the first task in a plan run.

    Returns:
        int: Exit code (0 = success, 5 = failed).
    """
    # Handle --print-prompt / --copy-prompt (static preview)
    if print_prompt or copy_prompt:
        # Initial build without retry context for preview
        prompt_text = builder.build(
            feature_dir,
            cur,
            include_phase0_reads=include_phase0_reads,
        )
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

    # Auto-QC phase detection: run toolchain loop without per-task LLM calls.
    auto_qc_phase: AutoQCPhase | None = None
    # Use a safe attribute lookup to support test doubles without methods.
    auto_qc_lookup = getattr(parser, "auto_qc_phase_for_task", None)
    if callable(auto_qc_lookup):
        candidate = auto_qc_lookup(cur)
        if isinstance(candidate, AutoQCPhase):
            auto_qc_phase = candidate
    if auto_qc_phase:
        return _execute_auto_qc_phase(
            workspace=workspace,
            phase=auto_qc_phase,
            parser=parser,
            qc_runner=qc_runner,
            log_file=log_file,
            feature_dir=feature_dir,
            preferred_model=preferred_model,
            run_id=run_id,
            copilot_rate_limiter=copilot_rate_limiter,
            copilot_backoff=copilot_backoff,
            copilot_max_retries=copilot_max_retries,
            copilot_output_tail_bytes=copilot_output_tail_bytes,
            copilot_allow_shell=copilot_allow_shell,
            copilot_allow_all_paths=copilot_allow_all_paths,
            copilot_allow_all_urls=copilot_allow_all_urls,
            copilot_trust_workspace=copilot_trust_workspace,
            max_fix_attempts=max_fix_attempts,
            print_prompt=print_prompt,
            copy_prompt=copy_prompt,
            is_first_task=is_first_task,
        )

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
        prompt_text = builder.build(
            feature_dir,
            cur,
            retry_context=retry_ctx,
            include_phase0_reads=include_phase0_reads,
        )

        limit_str = str(max_fix_attempts) if max_fix_attempts > 0 else "∞"
        msg = f"Executing task {cur.task_id} (attempt {attempt}/{limit_str})"
        print(msg)
        _log_msg(log_file, f"INFO: {msg}")

        copilot_invocation = 0
        throttle_retries = 0

        # Throttle-aware Copilot invocation loop.
        # - Rate-limit *call frequency* with the injected limiter.
        # - On throttle-like failures, apply bounded exponential backoff and retry.
        # - On non-throttle failures, fail fast with actionable context.
        while True:
            copilot_rate_limiter.acquire()

            copilot_result = run_copilot(
                workspace=workspace,
                prompt_text=prompt_text,
                log_file=log_file,
                task_id=cur.task_id,
                preferred_model=preferred_model,
                run_id=run_id,
                resume_session=(attempt > 1 or copilot_invocation > 0),
                is_first_task=is_first_task,
                allow_all_paths=copilot_allow_all_paths,
                allow_all_urls=copilot_allow_all_urls,
                allow_shell=copilot_allow_shell,
                trust_workspace=copilot_trust_workspace,
                _output_tail_bytes=copilot_output_tail_bytes,
            )
            copilot_invocation += 1

            if copilot_result.exit_code == 0:
                # A successful Copilot run clears any accumulated throttle state.
                copilot_backoff.on_success()
                break

            failure_kind = classify_copilot_failure(
                exit_code=copilot_result.exit_code,
                output_tail=copilot_result.output_tail,
            )
            if failure_kind is FailureKind.NON_THROTTLE:
                err_msg = (
                    "Copilot CLI failed (non-throttle). "
                    f"exit_code={copilot_result.exit_code}. "
                    f"output_tail={copilot_result.output_tail!r}"
                )
                print(err_msg, file=sys.stderr)
                _log_msg(log_file, f"ERROR: {err_msg}")
                print(f"See log: {log_file}", file=sys.stderr)
                return 5

            # Throttle-like failure: bounded retry with backoff.
            if copilot_max_retries < 0:
                copilot_max_retries = 0

            if throttle_retries >= copilot_max_retries:
                err_msg = (
                    f"Copilot CLI appears throttled, but max retries "
                    f"({copilot_max_retries}) were exhausted for task {cur.task_id}."
                )
                print(err_msg, file=sys.stderr)
                _log_msg(log_file, f"ERROR: {err_msg}")
                print(f"See log: {log_file}", file=sys.stderr)
                return 5

            delay_seconds = copilot_backoff.on_throttle()
            throttle_retries += 1

            retry_msg = (
                f"Copilot throttled for task {cur.task_id}; retry "
                f"{throttle_retries}/{copilot_max_retries} after "
                f"{delay_seconds:.2f}s backoff."
            )
            print(retry_msg)
            _log_msg(log_file, f"WARN: {retry_msg}")

            # Apply the backoff delay using the injected sleeper so tests can
            # remain deterministic (no real sleeps).
            if delay_seconds > 0:
                copilot_rate_limiter.sleeper.sleep(delay_seconds)

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

    lock_path: Path | None = None
    if args.cmd == "execute-all":
        lock_path = acquire_executor_lock(workspace)

    try:
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

        include_phase0_reads = False
        phase0_reads = _phase0_read_tasks(parser)
        if phase0_reads:
            include_phase0_reads = True

        # Bundle Phase 0 read tasks with the first non-read task on session start.
        if include_phase0_reads and _is_phase0_read_task(cur):
            non_read_task = _first_non_read_task(parser)
            if non_read_task is not None:
                cur = non_read_task

        builder = PromptBuilder(
            workspace,
            prompt_template_path,
            preferred_model=args.preferred_model,
        )
        qc_runner = QCRunner(workspace)

        # Per-run throttling controls. The limiter must persist across tasks to
        # regulate overall call cadence.
        copilot_rate_limiter = CallRateLimiter(
            max_calls=args.copilot_cli_max_calls_per_window,
            window_seconds=args.copilot_cli_window_seconds,
            clock=SystemClock(),
            sleeper=TimeSleeper(),
        )

        is_first_task = True

        while True:
            # Backoff state is per-task; it resets after successful Copilot invocations.
            copilot_backoff = ExponentialBackoff(
                base_seconds=args.copilot_cli_backoff_base_seconds,
                max_seconds=args.copilot_cli_backoff_max_seconds,
                random_source=SystemRandom(),
            )

            # Build prompt and execute
            result = execute_one_task(
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
                copilot_rate_limiter=copilot_rate_limiter,
                copilot_backoff=copilot_backoff,
                copilot_max_retries=args.copilot_cli_max_retries,
                copilot_output_tail_bytes=args.copilot_cli_output_tail_bytes,
                copilot_allow_shell=args.copilot_allow_shell,
                copilot_allow_all_paths=args.copilot_allow_all_paths,
                copilot_allow_all_urls=args.copilot_allow_all_urls,
                copilot_trust_workspace=args.copilot_trust_workspace,
                include_phase0_reads=include_phase0_reads and is_first_task,
                print_prompt=args.print_prompt,
                copy_prompt=args.copy_prompt,
                is_first_task=is_first_task,
            )

            if result != 0:
                return result

            # Stop here if interactive command (print/copy)
            if args.print_prompt or args.copy_prompt:
                return 0

            # Check phase completion after task success
            if parser.phase_complete(cur.phase):
                is_auto_qc = False
                # Avoid MagicMock truthiness by explicitly calling the checker.
                auto_qc_phase_check = getattr(parser, "is_auto_qc_phase", None)
                if callable(auto_qc_phase_check):
                    phase_candidate = auto_qc_phase_check(cur.phase)
                    if isinstance(phase_candidate, bool):
                        is_auto_qc = phase_candidate

                if is_auto_qc:
                    print(f"Phase {cur.phase} complete (auto-QC handled by executor).")
                else:
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
            is_first_task = False
            include_phase0_reads = False
            print(f"Proceeding to next task: {cur.task_id}...")
    finally:
        if lock_path is not None:
            release_executor_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

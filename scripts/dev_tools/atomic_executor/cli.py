"""
CLI entry point and orchestration for atomic executor.

Provides argument parsing, workspace validation, and main execution loop
that coordinates PlanParser, FeatureResolver, QCRunner, and PromptBuilder.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver
from scripts.dev_tools.atomic_executor.plan_parser import PlanParser, PlanTask
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_runner import QCRunner

DEFAULT_PROMPT_TEMPLATE = ".github/prompts/execute-atomic-plan.prompt.md"
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
    """
    result = subprocess.run(  # noqa: S603, S607 - trusted git cmd
        ["git", "status", "--porcelain"],  # noqa: S607
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
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
    try:
        result = subprocess.run(  # noqa: S603, S607 - trusted git cmd
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        b = result.stdout.strip()
        return b or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard using multiple fallback methods.

    Args:
        text (str): Text to copy.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Try pyperclip first
    try:
        import pyperclip  # type: ignore[import-untyped]

        pyperclip.copy(text)
        return True
    except (ImportError, Exception):  # noqa: S110 - expected fallback behavior
        # Suppress exceptions and fall through to command-line tools
        pass

    # Try platform-specific clipboard commands
    candidates: tuple[list[str], ...] = (
        ["pbcopy"],  # macOS
        ["wl-copy"],  # Wayland
        ["xclip", "-selection", "clipboard"],  # X11
        ["xsel", "--clipboard", "--input"],  # X11 alternative
        ["clip"],  # Windows
    )
    for cmd in candidates:
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            subprocess.run(  # noqa: S603 - exe resolved via shutil.which
                [exe, *cmd[1:]],
                input=text,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue

    return False


def run_copilot(
    *,
    workspace: Path,
    prompt_text: str,
    log_file: Path,
) -> None:
    """
    Invoke GitHub Copilot CLI with prompt and tool permissions.

    Args:
        workspace (Path): Repository root.
        prompt_text (str): Complete prompt to execute.
        log_file (Path): Path to log file for output.

    Raises:
        FileNotFoundError: If copilot executable not found.
        CalledProcessError: If copilot execution fails.

    Side Effects:
        - Executes copilot command
        - Writes to log file
    """
    copilot_exe = shutil.which("copilot")
    if not copilot_exe:
        raise FileNotFoundError("Required executable not found on PATH: copilot")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        copilot_exe,
        "-p",
        prompt_text,
        "--allow-tool",
        "write",
        "--allow-tool",
        "shell(poetry)",
        "--allow-tool",
        "shell(python)",
        "--allow-tool",
        "shell(git)",
    ]

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n\n=== Copilot invocation ===\n")
        f.write(
            "(prompt omitted from log for brevity; " "use --print-prompt to view)\n"
        )
        f.flush()
        subprocess.run(  # noqa: S603 - copilot_exe resolved via shutil.which
            argv, cwd=workspace, stdout=f, stderr=subprocess.STDOUT, check=True
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

        run_copilot(workspace=workspace, prompt_text=prompt_text, log_file=log_file)

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


def main(argv: list[str]) -> int:
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
    args = parse_args(argv)
    workspace = resolve_workspace(args.workspace)

    # Preconditions: clean tree, not on protected branch
    ensure_clean_tree(workspace)
    refuse_protected_branch(workspace)

    # Resolve feature folder
    active_dir = workspace / "docs" / "features" / "active"
    resolver = FeatureResolver(workspace, active_dir)
    _, feature_dir = resolver.resolve(args.path, args.feature)

    plan_path = feature_dir / "plan.md"
    prompt_template_path = (workspace / args.prompt_template).resolve()

    if not plan_path.is_file():
        print(f"Missing required plan.md: {plan_path}", file=sys.stderr)
        return 2
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

    builder = PromptBuilder(workspace, prompt_template_path)
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

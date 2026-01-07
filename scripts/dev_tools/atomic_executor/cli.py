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
from scripts.dev_tools.atomic_executor.plan_parser import PlanParser
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
    run_id = subprocess.run(  # noqa: S603, S607 - trusted date cmd
        ["date", "+%Y-%m-%d_%H%M%S"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
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
    else:
        if args.start:
            cur = parser.find_task_by_id(args.start)
        else:
            cur = parser.next_unchecked_task()
            if cur is None:
                print("Plan already complete: no unchecked tasks found.")
                return 0

    # Build prompt
    builder = PromptBuilder(workspace, prompt_template_path)
    prompt_text = builder.build(feature_dir, cur)

    # Handle --print-prompt / --copy-prompt
    if args.print_prompt:
        print(prompt_text)
        return 0

    if args.copy_prompt:
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

    # Execute exactly one task per run
    qc_runner = QCRunner(workspace)

    for attempt in range(1, args.max_fix_attempts + 1):
        print(
            f"Executing task {cur.task_id} "
            f"(attempt {attempt}/{args.max_fix_attempts})"
        )
        run_copilot(workspace=workspace, prompt_text=prompt_text, log_file=log_file)

        # Refresh plan after copilot run
        parser_after = PlanParser(plan_path)
        cur_after = parser_after.find_task_by_id(cur.task_id)

        # Task-step QC (scoped)
        try:
            qc_runner.run_scoped()
        except subprocess.CalledProcessError as e:
            print(
                f"Scoped QC failed for task {cur.task_id}: {e}",
                file=sys.stderr,
            )
            continue

        # Flip checkbox if model didn't do it (authoritative edit after QC)
        if not cur_after.checked:
            parser.flip_checkbox(cur)

        # Check phase completion
        parser_now = PlanParser(plan_path)
        if parser_now.phase_complete(cur.phase):
            print(f"Phase {cur.phase} complete -> running full toolchain...")
            try:
                qc_runner.run_full()
            except subprocess.CalledProcessError as e:
                print(
                    f"Full QC failed after completing Phase {cur.phase}: {e}",
                    file=sys.stderr,
                )
                # Phase QC failure: user should decide revert/adjust
                return 5

        print(
            f"Task {cur.task_id} complete and gated. "
            f"Next: run 'resume' for the next task."
        )
        return 0

    # Max attempts exhausted
    print(
        f"Failed to complete task {cur.task_id} after "
        f"{args.max_fix_attempts} attempts.",
        file=sys.stderr,
    )
    print(f"See log: {log_file}", file=sys.stderr)
    return 5


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

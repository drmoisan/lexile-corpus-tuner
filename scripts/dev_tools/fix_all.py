"""Python implementation of the fix-all workflow."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from scripts.dev_tools import fix_all_runner
from scripts.dev_tools.fix_all_runner import (
    BranchResult,
    CommandResult,
    CommandRunner,
    StepLogger,
    SubprocessCommandRunner,
    run_fix_all,
    shell_test_was_skipped,
    subprocess_run,
)
from scripts.dev_tools.fix_all_status_board import (
    format_ansi_redraw,
    format_status_transition_line,
    is_vt_enabled_for_stream,
    render_status_board,
    should_use_interactive_board,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BranchResult",
    "CommandResult",
    "CommandRunner",
    "StepLogger",
    "SubprocessCommandRunner",
    "format_status_transition_line",
    "render_status_board",
    "format_ansi_redraw",
    "should_use_interactive_board",
    "is_vt_enabled_for_stream",
    "shell_test_was_skipped",
    "run_fix_all",
    "parse_args",
    "main",
    "_shell_test_was_skipped",
    "fix_all_runner",
    "subprocess_run",
    "sys",
]


def _shell_test_was_skipped(output: str) -> bool:
    """Backward-compatible wrapper for shell skip detection."""
    return shell_test_was_skipped(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the fix-all workflow.

    Purpose:
        Define and parse CLI options for the fix-all execution entry point.

    Args:
        argv (Sequence[str] | None): Optional argument list for testing or CLI use.

    Returns:
        argparse.Namespace: Parsed arguments with configured defaults.

    Raises:
        SystemExit: Raised by argparse when parsing fails or help is requested.

    Side Effects:
        Writes help or error text to stdout/stderr via argparse when applicable.
    """
    parser = argparse.ArgumentParser(
        description="Run all code quality steps with auto-fix and retries."
    )
    parser.add_argument(
        "--complete-all",
        action="store_true",
        help="Run all branches to completion even if another branch fails.",
    )
    parser.add_argument(
        "--max-ruff-retries",
        type=int,
        default=3,
        help="Maximum number of Ruff --fix retries (default: 3).",
    )
    parser.add_argument(
        "--max-black-retries",
        type=int,
        default=3,
        help="Maximum number of Black retries (default: 3).",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage flags when running pytest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the fix-all workflow using CLI arguments.

    Purpose:
        Provide the command-line entry point for running fix-all.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments for testing or CLI use.

    Returns:
        int: Process exit code (0 for success, 1 for failure).

    Raises:
        SystemExit: Raised by argparse if parsing fails or help is requested.

    Side Effects:
        Executes the fix-all pipeline and writes output to stdout.
    """
    args = parse_args(argv)
    return run_fix_all(
        max_ruff_retries=args.max_ruff_retries,
        max_black_retries=args.max_black_retries,
        include_coverage=not args.no_coverage,
        complete_all=args.complete_all,
    )


if __name__ == "__main__":
    raise SystemExit(main())

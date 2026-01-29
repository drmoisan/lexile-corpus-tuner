"""
Argument parsing for atomic executor CLI.

Purpose:
    Provides CLI argument parsing with common options shared across
    execute, resume, and execute-all subcommands.

Usage:
    from scripts.dev_tools.atomic_executor.arg_parser import parse_args
    args = parse_args(sys.argv[1:])
"""

from __future__ import annotations

import argparse

# Constants for argument defaults (moved from cli to avoid circular import)
DEFAULT_PROMPT_TEMPLATE = ".github/prompts/execute-plan-template.md"
DEFAULT_COPILOT_CLI_MAX_CALLS_PER_WINDOW = 6
DEFAULT_COPILOT_CLI_WINDOW_SECONDS = 60.0
DEFAULT_COPILOT_CLI_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_COPILOT_CLI_BACKOFF_MAX_SECONDS = 60.0
DEFAULT_COPILOT_CLI_OUTPUT_TAIL_BYTES = 4096
DEFAULT_COPILOT_CLI_MAX_RETRIES = 8
DEFAULT_COPILOT_ALLOW_SHELL = True
DEFAULT_COPILOT_ALLOW_ALL_PATHS = True
DEFAULT_COPILOT_ALLOW_ALL_URLS = False
DEFAULT_COPILOT_TRUST_WORKSPACE = True


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Purpose:
        Parses command-line arguments for the atomic executor, supporting
        three subcommands: execute, resume, and execute-all.

    Args:
        argv (list[str]): Command-line arguments (typically sys.argv[1:]).

    Returns:
        argparse.Namespace: Parsed arguments with fields like cmd, path,
            workspace, feature, prompt_template, start, max_fix_attempts,
            print_prompt, copy_prompt, preferred_model, and various
            copilot-cli-* throttling controls.
    """
    p = argparse.ArgumentParser(description="Atomic task-by-task executor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        """
        Add common arguments to a subcommand parser.

        Purpose:
            Defines the shared argument structure for execute, resume,
            and execute-all subcommands to avoid duplication.

        Args:
            sp (argparse.ArgumentParser): Subparser to add arguments to.

        Side Effects:
            Modifies sp by adding argument definitions.
        """
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
        sp.add_argument(
            "--skip-preflight-qc",
            action="store_true",
            default=False,
            help=(
                "Skip the pre-flight QC check that runs before task execution. "
                "By default, execute-all runs a full QC and invokes Copilot to fix "
                "any baseline failures before proceeding."
            ),
        )

    sp_exec = sub.add_parser("execute", help="Execute from first unchecked or --start.")
    add_common(sp_exec)

    sp_resume = sub.add_parser("resume", help="Resume from first unchecked task.")
    add_common(sp_resume)

    sp_all = sub.add_parser("execute-all", help="Execute all remaining tasks.")
    add_common(sp_all)

    return p.parse_args(argv)

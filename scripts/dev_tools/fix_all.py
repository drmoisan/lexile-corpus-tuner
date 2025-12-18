"""Python implementation of the fix-all workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class CommandResult:
    """Result of a command invocation."""

    returncode: int
    output: str


class CommandRunner(Protocol):
    """Protocol for running commands within the fix-all pipeline."""

    def run(self, command: Sequence[str], *, step_name: str) -> CommandResult:
        """Execute the provided command and return the result."""
        ...


@dataclass
class StepLogger:
    """Simple logger for emitting step, success, and failure messages."""

    stream: TextIO = sys.stdout

    def step(self, message: str) -> None:
        print(f"\n==> {message}", file=self.stream)

    def success(self, message: str) -> None:
        print(f"[OK] {message}", file=self.stream)

    def failure(self, message: str) -> None:
        print(f"[FAIL] {message}", file=self.stream)

    def info(self, message: str) -> None:
        print(message, file=self.stream)

    def command_output(self, output: str) -> None:
        if output:
            end = "" if output.endswith("\n") else "\n"
            print(output, file=self.stream, end=end)

    def separator(self) -> None:
        print("", file=self.stream)


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    return "".join(parts)


subprocess_run = subprocess.run


@dataclass
class SubprocessCommandRunner:
    """Real command runner that invokes subprocesses."""

    logger: StepLogger

    def run(self, command: Sequence[str], *, step_name: str) -> CommandResult:
        if not command:
            raise ValueError("command cannot be empty.")

        result = subprocess_run(  # noqa: S603
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
        output = _combine_output(result.stdout, result.stderr)
        if output:
            self.logger.command_output(output)
        return CommandResult(returncode=result.returncode, output=output)


def _log_failure(logger: StepLogger, message: str, result: CommandResult) -> None:
    logger.failure(f"{message} (exit code {result.returncode})")
    if result.output:
        logger.failure("Command output:")
        logger.command_output(result.output)


def _run_simple_step(
    *,
    step_number: int,
    description: str,
    step_name: str,
    success_message: str,
    failure_message: str,
    command: Sequence[str],
    runner: CommandRunner,
    logger: StepLogger,
) -> bool:
    logger.step(f"Step {step_number}: {description}")
    result = runner.run(command, step_name=step_name)
    if result.returncode == 0:
        logger.success(success_message)
        return True

    _log_failure(logger, failure_message, result)
    return False


def _ruff_fix(
    *,
    max_retries: int,
    runner: CommandRunner,
    logger: StepLogger,
) -> bool:
    logger.step("Step 2: Running Ruff linting with auto-fix...")
    attempt = 0
    last_result: CommandResult | None = None
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Ruff attempt {attempt} of {max_retries}...")
        result = runner.run(
            ["poetry", "run", "ruff", "check", "--fix"], step_name="Ruff: fix"
        )
        last_result = result
        if result.returncode == 0:
            logger.success("Ruff linting passed")
            return True

        if attempt < max_retries:
            logger.info("Ruff found issues. Retrying...")
    _log_failure(
        logger,
        (
            f"Ruff linting failed after {max_retries} attempts. "
            "Please review errors above."
        ),
        last_result or CommandResult(returncode=1, output=""),
    )
    return False


def run_fix_all(
    *,
    max_ruff_retries: int = 3,
    include_coverage: bool = True,
    runner: CommandRunner | None = None,
    logger: StepLogger | None = None,
) -> int:
    """Run the fix-all pipeline, returning 0 on success and 1 on failure."""
    step_logger = logger or StepLogger()
    command_runner = runner or SubprocessCommandRunner(step_logger)

    if not _run_simple_step(
        step_number=1,
        description="Running Black formatting...",
        step_name="Black: format",
        success_message="Black formatting completed successfully",
        failure_message="Black formatting failed. Please review errors above.",
        command=["poetry", "run", "black", "."],
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    if not _ruff_fix(
        max_retries=max_ruff_retries,
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    if not _run_simple_step(
        step_number=3,
        description="Re-running Black to ensure consistency...",
        step_name="Black: format (verify)",
        success_message="Black formatting verified",
        failure_message="Black formatting failed on verification pass.",
        command=["poetry", "run", "black", "."],
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    if not _run_simple_step(
        step_number=4,
        description="Re-running Ruff to verify fixes...",
        step_name="Ruff: lint (verify)",
        success_message="Ruff linting verified",
        failure_message=(
            "Ruff linting still has issues after fixes. Please review errors above."
        ),
        command=["poetry", "run", "ruff", "check"],
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    if not _run_simple_step(
        step_number=5,
        description="Running Pyright type checking...",
        step_name="Pyright: type-check",
        success_message="Pyright type checking passed",
        failure_message="Pyright type checking failed. Please review errors above.",
        command=["poetry", "run", "pyright"],
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    pytest_command: list[str] = ["poetry", "run", "pytest"]
    pytest_step_name = (
        "Pytest: test with coverage" if include_coverage else "Pytest: test"
    )
    if include_coverage:
        pytest_command.extend(
            [
                "--cov=src/lexile_corpus_tuner",
                "--cov=scripts/dev_tools",
                "--cov-report=term-missing",
            ]
        )

    if not _run_simple_step(
        step_number=6,
        description=(
            "Running Pytest with coverage..."
            if include_coverage
            else "Running Pytest..."
        ),
        step_name=pytest_step_name,
        success_message="Pytest passed",
        failure_message="Pytest failed. Please review errors above.",
        command=pytest_command,
        runner=command_runner,
        logger=step_logger,
    ):
        return 1

    step_logger.separator()
    step_logger.info("========================================")
    step_logger.info("ALL CHECKS PASSED")
    step_logger.info("========================================")
    step_logger.info("  Black formatting: PASS")
    step_logger.info("  Ruff linting: PASS")
    step_logger.info("  Pyright type checking: PASS")
    step_logger.info(
        "  Pytest with coverage: PASS" if include_coverage else "  Pytest: PASS"
    )
    step_logger.info("========================================")

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all code quality steps with auto-fix and retries."
    )
    parser.add_argument(
        "--max-ruff-retries",
        type=int,
        default=3,
        help="Maximum number of Ruff --fix retries (default: 3).",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage flags when running pytest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_fix_all(
        max_ruff_retries=args.max_ruff_retries, include_coverage=not args.no_coverage
    )


if __name__ == "__main__":
    raise SystemExit(main())

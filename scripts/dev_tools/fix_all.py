"""Python implementation of the fix-all workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class CommandResult:
    """Result of a command invocation."""

    returncode: int
    output: str


@dataclass
class BranchResult:
    """Outcome for a single toolchain branch."""

    name: str
    success: bool
    output: str
    failed_step: str | None = None


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
    attempt = 0
    last_result: CommandResult | None = None
    # Retry Ruff --fix so lint errors have multiple chances to be auto-corrected.
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Ruff --fix attempt {attempt} of {max_retries}...")
        result = runner.run(
            ["poetry", "run", "ruff", "check", "--fix"], step_name="Ruff: fix"
        )
        last_result = result
        if result.returncode == 0:
            logger.success("Ruff auto-fix completed")
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


def _run_black_with_retry(
    *,
    step_number: int,
    max_retries: int,
    runner: CommandRunner,
    logger: StepLogger,
) -> bool:
    attempt = 0
    # Black is retried to allow it to reformat files after upstream tools apply fixes.
    while attempt < max_retries:
        attempt += 1
        logger.step(
            f"Step {step_number}: Running Black formatting... "
            f"(attempt {attempt} of {max_retries})"
        )
        result = runner.run(["poetry", "run", "black", "."], step_name="Black: format")
        if result.returncode == 0:
            logger.success("Black formatting completed successfully")
            return True

        if attempt < max_retries:
            logger.info("Black found issues. Retrying...")
        else:
            _log_failure(
                logger,
                f"Black formatting failed after {max_retries} attempts.",
                result,
            )
    return False


def run_fix_all(
    *,
    max_ruff_retries: int = 3,
    max_black_retries: int = 3,
    include_coverage: bool = True,
    runner_factory: Callable[[str, StepLogger], CommandRunner] | None = None,
    logger: StepLogger | None = None,
) -> int:
    """Run the fix-all pipeline in parallel branches.

    Branches (JSON, shell, Python, PowerShell) execute concurrently; failure in one
    branch does not stop others. The final exit code is 0 only if all branches pass.
    """

    step_logger = logger or StepLogger()

    def factory(branch_name: str, branch_logger: StepLogger) -> CommandRunner:
        if runner_factory is not None:
            return runner_factory(branch_name, branch_logger)
        return SubprocessCommandRunner(branch_logger)

    def run_json_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("json", branch_logger)

        if not _run_simple_step(
            step_number=1,
            description="Running JSON formatting...",
            step_name="JSON: format",
            success_message="JSON formatting completed",
            failure_message="JSON formatting failed. Please review errors above.",
            command=[
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.format_json",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="json", success=False, output=output, failed_step="JSON: format"
            )

        if not _run_simple_step(
            step_number=2,
            description="Running JSON validation...",
            step_name="JSON: validate",
            success_message="JSON validation passed",
            failure_message="JSON validation failed. Please review errors above.",
            command=[
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.validate_json",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="json", success=False, output=output, failed_step="JSON: validate"
            )

        output = branch_stream.getvalue()
        return BranchResult(name="json", success=True, output=output)

    def run_shell_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("shell", branch_logger)

        if not _run_simple_step(
            step_number=1,
            description="Running shell script formatting (shfmt)...",
            step_name="Shell: format",
            success_message="Shell formatting completed",
            failure_message="Shell formatting failed. Please review errors above.",
            command=[
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.shell_qc",
                "format",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="shell", success=False, output=output, failed_step="Shell: format"
            )

        if not _run_simple_step(
            step_number=2,
            description="Running shell linting (shfmt -d + shellcheck)...",
            step_name="Shell: check",
            success_message="Shell linting passed",
            failure_message="Shell linting failed. Please review errors above.",
            command=[
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.shell_qc",
                "check",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="shell", success=False, output=output, failed_step="Shell: check"
            )

        if not _run_simple_step(
            step_number=3,
            description="Running shell tests (bats)...",
            step_name="Shell: test",
            success_message="Shell tests passed",
            failure_message="Shell tests failed. Please review errors above.",
            command=[
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.shell_qc",
                "test",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="shell", success=False, output=output, failed_step="Shell: test"
            )

        output = branch_stream.getvalue()
        return BranchResult(name="shell", success=True, output=output)

    def run_python_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("python", branch_logger)

        # Restart Black→Ruff when Ruff fixes files to keep ordering consistent.
        while True:
            if not _run_black_with_retry(
                step_number=1,
                max_retries=max_black_retries,
                runner=branch_runner,
                logger=branch_logger,
            ):
                output = branch_stream.getvalue()
                return BranchResult(
                    name="python",
                    success=False,
                    output=output,
                    failed_step="Black: format",
                )

            branch_logger.step("Step 2: Running Ruff linting...")
            ruff_result = branch_runner.run(
                ["poetry", "run", "ruff", "check"], step_name="Ruff: lint"
            )
            if ruff_result.returncode == 0:
                branch_logger.success("Ruff linting passed")
            else:
                if ruff_result.output:
                    branch_logger.command_output(ruff_result.output)
                branch_logger.info("Ruff reported issues; attempting auto-fix...")
                if not _ruff_fix(
                    max_retries=max_ruff_retries,
                    runner=branch_runner,
                    logger=branch_logger,
                ):
                    output = branch_stream.getvalue()
                    return BranchResult(
                        name="python",
                        success=False,
                        output=output,
                        failed_step="Ruff: lint",
                    )
                branch_logger.info(
                    "Ruff auto-fix applied; restarting Black to re-verify formatting."
                )
                branch_logger.info("Re-running Black and Ruff to confirm clean state.")
                continue

            break

        if not _run_simple_step(
            step_number=3,
            description="Running Pyright type checking...",
            step_name="Pyright: type-check",
            success_message="Pyright type checking passed",
            failure_message="Pyright type checking failed. Please review errors above.",
            command=["poetry", "run", "pyright"],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="python",
                success=False,
                output=output,
                failed_step="Pyright: type-check",
            )

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
            step_number=4,
            description=(
                "Running Pytest with coverage..."
                if include_coverage
                else "Running Pytest..."
            ),
            step_name=pytest_step_name,
            success_message="Pytest passed",
            failure_message="Pytest failed. Please review errors above.",
            command=pytest_command,
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="python",
                success=False,
                output=output,
                failed_step=pytest_step_name,
            )

        output = branch_stream.getvalue()
        return BranchResult(name="python", success=True, output=output)

    def run_powershell_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("powershell", branch_logger)

        if not _run_simple_step(
            step_number=1,
            description="Running PowerShell formatting (Invoke-PoshQCFormat)...",
            step_name="PoshQC: format",
            success_message="PowerShell formatting completed",
            failure_message="PowerShell formatting failed. Please review errors above.",
            command=[
                "pwsh",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Import-Module './scripts/powershell/PoshQC'; "
                "Invoke-PoshQCFormat -Root '.'",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: format",
            )

        if not _run_simple_step(
            step_number=2,
            description="Running PowerShell linting (PSScriptAnalyzer)...",
            step_name="PoshQC: analyze",
            success_message="PowerShell analysis passed",
            failure_message="PowerShell analysis failed. Please review errors above.",
            command=[
                "pwsh",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Import-Module './scripts/powershell/PoshQC'; "
                "Invoke-PoshQCAnalyze -Root '.'",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: analyze",
            )

        if not _run_simple_step(
            step_number=3,
            description="Running PowerShell tests (Pester)...",
            step_name="PoshQC: test",
            success_message="PowerShell tests passed",
            failure_message="PowerShell tests failed. Please review errors above.",
            command=[
                "pwsh",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Import-Module './scripts/powershell/PoshQC'; "
                "Invoke-PoshQCTest -Root '.'",
            ],
            runner=branch_runner,
            logger=branch_logger,
        ):
            output = branch_stream.getvalue()
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: test",
            )

        output = branch_stream.getvalue()
        return BranchResult(name="powershell", success=True, output=output)

    branch_functions: list[tuple[str, Callable[[], BranchResult]]] = [
        ("json", run_json_branch),
        ("shell", run_shell_branch),
        ("python", run_python_branch),
        ("powershell", run_powershell_branch),
    ]

    results: dict[str, BranchResult] = {}
    threads: list[threading.Thread] = []

    def _runner(name: str, func: Callable[[], BranchResult]) -> None:
        results[name] = func()

    for name, func in branch_functions:
        thread = threading.Thread(target=_runner, args=(name, func), daemon=True)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for name, _ in branch_functions:
        branch_result = results.get(name)
        if branch_result is None:
            continue
        step_logger.separator()
        step_logger.info(f"--- {name} branch log ---")
        if branch_result.output:
            step_logger.info(branch_result.output)
        else:
            step_logger.info("(no output)")

    step_logger.separator()
    step_logger.info("========== Branch Results ==========")
    for name, _ in branch_functions:
        branch_result = results.get(name)
        if branch_result is None:
            step_logger.failure(f"Branch {name} did not produce a result.")
            continue

        status = "PASS" if branch_result.success else "FAIL"
        if branch_result.failed_step:
            step_logger.info(
                f"Branch {name}: {status} (failed at {branch_result.failed_step})"
            )
        else:
            step_logger.info(f"Branch {name}: {status}")
    step_logger.info("====================================")

    return 0 if all(res.success for res in results.values()) else 1


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
    args = parse_args(argv)
    return run_fix_all(
        max_ruff_retries=args.max_ruff_retries,
        max_black_retries=args.max_black_retries,
        include_coverage=not args.no_coverage,
    )


if __name__ == "__main__":
    raise SystemExit(main())

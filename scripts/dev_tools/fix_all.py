"""Python implementation of the fix-all workflow."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import threading
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Protocol, TextIO, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
]


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


class Kernel32Api(Protocol):
    """Typed protocol for the subset of Kernel32 APIs used for VT enablement."""

    def GetStdHandle(self, n_std_handle: int) -> int: ...

    def GetConsoleMode(self, handle: int, mode: object) -> int: ...

    def SetConsoleMode(self, handle: int, mode: int) -> int: ...


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


def format_status_transition_line(branch: str, status: str) -> str:
    """
    Format a non-interactive status transition line.

    Purpose:
        Provide a deterministic, line-oriented status update for CI or redirected
        output streams.

    Args:
        branch (str): Branch name to include in the output line.
        status (str): Status string to include in the output line.

    Returns:
        str: A formatted status line using the required STATUS|... template.

    Raises:
        ValueError: Raised if branch or status is empty.

    Side Effects:
        None. Pure formatting function.
    """
    if not branch:
        raise ValueError("branch cannot be empty.")
    if not status:
        raise ValueError("status cannot be empty.")
    return f"STATUS|branch={branch}|status={status}"


def render_status_board(lines: list[str], *, width: int) -> str:
    """
    Render a fixed-height status board for interactive terminals.

    Purpose:
        Produce deterministic board text with one line per branch for in-place
        redraws in interactive terminals.

    Args:
        lines (list[str]): Preformatted status lines to render.
        width (int): Target board width for padding or truncation decisions.

    Returns:
        str: Rendered board text with one newline per line and a trailing newline.

    Raises:
        ValueError: Raised when width is not positive.

    Side Effects:
        None. Pure rendering function.
    """
    if width <= 0:
        raise ValueError("width must be positive.")

    if not lines:
        # Return empty output to avoid trailing newline for empty boards.
        return ""

    rendered_lines: list[str] = []
    # Pad or trim each line to keep the board width stable between redraws.
    for line in lines:
        if len(line) > width:
            rendered_lines.append(line[:width])
        else:
            rendered_lines.append(line.ljust(width))
    return "\n".join(rendered_lines) + "\n"


def format_ansi_redraw(board: str, *, line_count: int) -> str:
    """
    Format an ANSI redraw payload using erase-line and cursor-up sequences.

    Purpose:
        Build a deterministic ANSI redraw string that rewrites a fixed-height
        status board without emitting unsupported control sequences.

    Args:
        board (str): Rendered board content to write.
        line_count (int): Number of lines in the board to move the cursor up.

    Returns:
        str: ANSI redraw payload using only erase-line and cursor-up sequences.

    Raises:
        ValueError: Raised when line_count is negative.

    Side Effects:
        None. Pure formatting function.
    """
    if line_count < 0:
        raise ValueError("line_count cannot be negative.")

    output_parts: list[str] = []
    if line_count:
        output_parts.append("\x1b[1A" * line_count)
    # Clear each line before writing to avoid leftover characters from prior redraws.
    for line in board.splitlines():
        output_parts.append(f"\x1b[2K\r{line}\n")
    return "".join(output_parts)


def should_use_interactive_board(*, isatty: bool, vt_enabled: bool) -> bool:
    """
    Decide whether interactive status rendering should be used.

    Purpose:
        Gate terminal redraw behavior on TTY availability and VT support.

    Args:
        isatty (bool): Whether the output stream is a TTY.
        vt_enabled (bool): Whether VT/ANSI sequences are supported.

    Returns:
        bool: True when interactive rendering should be enabled.

    Raises:
        None.

    Side Effects:
        None. Pure decision function.
    """
    return isatty and vt_enabled


def _stream_isatty(stream: TextIO) -> bool:
    """
    Safely determine whether a stream is attached to a TTY.

    Purpose:
        Provide a defensive check for interactive output when streams may not
        implement the isatty method (e.g., StringIO).

    Args:
        stream (TextIO): Stream to query for TTY capability.

    Returns:
        bool: True when the stream reports itself as a TTY.

    Raises:
        None.

    Side Effects:
        None. Pure detection helper.
    """
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    return bool(isatty())


def is_vt_enabled_for_stream(stream: TextIO) -> bool:
    """
    Determine whether VT/ANSI support is enabled for the provided stream.

    Purpose:
        Enable Windows VT processing when possible and report whether ANSI
        sequences should be used for interactive rendering.

    Args:
        stream (TextIO): Stream to evaluate for VT support.

    Returns:
        bool: True when VT/ANSI sequences are supported for the stream.

    Raises:
        None.

    Side Effects:
        On Windows, attempts to enable VT processing for the console handle.
    """
    if not sys.platform.startswith("win"):
        return True

    from ctypes import wintypes

    enable_virtual_terminal_processing = 0x0004
    enable_processed_output = 0x0001
    std_output_handle = -11

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False

    kernel32 = cast(Kernel32Api, windll.kernel32)
    handle = kernel32.GetStdHandle(std_output_handle)
    if handle in (0, -1):
        return False

    mode = wintypes.DWORD()
    # On Windows, enable VT processing when a console mode is available.
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
        return False

    new_mode = mode.value | enable_virtual_terminal_processing | enable_processed_output
    if kernel32.SetConsoleMode(handle, new_mode) == 0:
        return False
    return True


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    return "".join(parts)


def shell_test_was_skipped(output: str) -> bool:
    """
    Detect whether shell tests were skipped based on output text.

    Purpose:
        Distinguish a successful shell test run from a skipped run so the status
        board can surface "SKIP tests" instead of a normal pass.

    Args:
        output (str): Combined stdout/stderr from the shell test step.

    Returns:
        bool: True when the output indicates shell tests were skipped.

    Raises:
        None.

    Side Effects:
        None. Pure detection helper.
    """
    skip_markers = (
        "No shell test directories found; skipping.",
        "bats not installed; skipping shell tests.",
    )
    # Check for any known skip marker emitted by the shell QC tooling.
    return any(marker in output for marker in skip_markers)


def _shell_test_was_skipped(output: str) -> bool:
    """
    Backward-compatible wrapper for shell skip detection.

    Purpose:
        Preserve the private helper name while delegating to the public
        shell skip detection implementation.

    Args:
        output (str): Combined stdout/stderr from the shell test step.

    Returns:
        bool: True when the output indicates shell tests were skipped.

    Raises:
        None.

    Side Effects:
        None. Pure detection helper.
    """
    return shell_test_was_skipped(output)


subprocess_run = subprocess.run

# Brief delay to allow fail-fast cancellation signals between step boundaries.
CANCEL_CHECK_DELAY_S: float = 0.01


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
    complete_all: bool = False,
) -> int:
    """
    Run the fix-all pipeline in parallel branches.

    Purpose:
        Execute JSON, shell, Python, and PowerShell quality checks concurrently
        while coordinating fail-fast behavior when configured.

    Args:
        max_ruff_retries (int): Maximum retry attempts for Ruff auto-fix.
        max_black_retries (int): Maximum retry attempts for Black formatting.
        include_coverage (bool): Whether pytest should include coverage flags.
        runner_factory (Callable[[str, StepLogger], CommandRunner] | None):
            Optional runner factory for testing or custom execution.
        logger (StepLogger | None): Optional shared logger for final output.
        complete_all (bool): When True, run all branches to completion even if
            another branch fails.

    Returns:
        int: Exit code (0 for success, 1 for any branch failure).

    Raises:
        ValueError: Raised when a command runner receives an empty command.

    Side Effects:
        Spawns threads, writes branch logs to buffers, and prints summary output.
    """

    step_logger = logger or StepLogger()
    cancel_event = threading.Event()
    stream_isatty = _stream_isatty(step_logger.stream)
    use_interactive_board = should_use_interactive_board(
        isatty=stream_isatty,
        vt_enabled=is_vt_enabled_for_stream(step_logger.stream),
    )
    status_lock = threading.Lock()
    status_by_branch = {
        "json": "pending",
        "shell": "pending",
        "python": "pending",
        "powershell": "pending",
    }
    has_rendered_board = False

    def emit_status_transition(branch: str, status: str) -> None:
        """
        Emit a status transition line when interactive rendering is disabled.

        Purpose:
            Keep CI and redirected logs readable by writing line-oriented status
            updates at step boundaries.

        Args:
            branch (str): Branch name emitting the status update.
            status (str): Status text for the transition.

        Returns:
            None.

        Raises:
            ValueError: Raised if branch or status is empty.

        Side Effects:
            Writes status lines to the main logger stream when non-interactive.
        """
        nonlocal has_rendered_board

        # Choose ANSI redraws for interactive terminals and line output otherwise.
        if use_interactive_board:
            with status_lock:
                status_by_branch[branch] = status
                # Build lines in a fixed order to keep the board stable between redraws.
                lines = [
                    f"{name}: {status_by_branch[name]}"
                    for name in ("json", "shell", "python", "powershell")
                ]
                width = max(len(line) for line in lines) if lines else 1
                board = render_status_board(lines, width=width)
                line_count = len(lines) if has_rendered_board else 0
                redraw = format_ansi_redraw(board, line_count=line_count)
                step_logger.stream.write(redraw)
                step_logger.stream.flush()
                has_rendered_board = True
            return

        line = format_status_transition_line(branch, status)
        print(line, file=step_logger.stream)

    def factory(branch_name: str, branch_logger: StepLogger) -> CommandRunner:
        if runner_factory is not None:
            return runner_factory(branch_name, branch_logger)
        return SubprocessCommandRunner(branch_logger)

    def run_json_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("json", branch_logger)

        emit_status_transition("json", "JSON: format")
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
            emit_status_transition("json", "FAIL")
            return BranchResult(
                name="json", success=False, output=output, failed_step="JSON: format"
            )

        if cancel_event.is_set() and not complete_all:
            output = branch_stream.getvalue()
            emit_status_transition("json", "FAIL")
            return BranchResult(
                name="json", success=False, output=output, failed_step="Canceled"
            )

        if not complete_all:
            # Allow fail-fast cancellation signals to arrive at step boundaries.
            cancel_event.wait(CANCEL_CHECK_DELAY_S)
        if cancel_event.is_set() and not complete_all:
            output = branch_stream.getvalue()
            emit_status_transition("json", "FAIL")
            return BranchResult(
                name="json", success=False, output=output, failed_step="Canceled"
            )

        emit_status_transition("json", "JSON: validate")
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
            emit_status_transition("json", "FAIL")
            return BranchResult(
                name="json", success=False, output=output, failed_step="JSON: validate"
            )

        output = branch_stream.getvalue()
        emit_status_transition("json", "PASS")
        return BranchResult(name="json", success=True, output=output)

    def run_shell_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("shell", branch_logger)

        emit_status_transition("shell", "Shell: format")
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
            emit_status_transition("shell", "FAIL")
            return BranchResult(
                name="shell", success=False, output=output, failed_step="Shell: format"
            )

        emit_status_transition("shell", "Shell: check")
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
            emit_status_transition("shell", "FAIL")
            return BranchResult(
                name="shell", success=False, output=output, failed_step="Shell: check"
            )

        emit_status_transition("shell", "Shell: test")
        branch_logger.step("Step 3: Running shell tests (bats)...")
        test_result = branch_runner.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "scripts.dev_tools.shell_qc",
                "test",
            ],
            step_name="Shell: test",
        )
        if test_result.returncode == 0:
            # Identify skipped tests so the status board can surface it explicitly.
            if shell_test_was_skipped(test_result.output):
                branch_logger.success("Shell tests skipped")
                test_status = "SKIP tests"
            else:
                branch_logger.success("Shell tests passed")
                test_status = "PASS"
            output = branch_stream.getvalue()
            emit_status_transition("shell", test_status)
            return BranchResult(name="shell", success=True, output=output)

        _log_failure(
            branch_logger,
            "Shell tests failed. Please review errors above.",
            test_result,
        )
        output = branch_stream.getvalue()
        emit_status_transition("shell", "FAIL")
        return BranchResult(
            name="shell", success=False, output=output, failed_step="Shell: test"
        )

    def run_python_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("python", branch_logger)

        # Restart Black→Ruff when Ruff fixes files to keep ordering consistent.
        while True:
            emit_status_transition("python", "Black: format")
            if not _run_black_with_retry(
                step_number=1,
                max_retries=max_black_retries,
                runner=branch_runner,
                logger=branch_logger,
            ):
                output = branch_stream.getvalue()
                emit_status_transition("python", "FAIL")
                return BranchResult(
                    name="python",
                    success=False,
                    output=output,
                    failed_step="Black: format",
                )

            emit_status_transition("python", "Ruff: lint")
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
                emit_status_transition("python", "Ruff: fix")
                if not _ruff_fix(
                    max_retries=max_ruff_retries,
                    runner=branch_runner,
                    logger=branch_logger,
                ):
                    output = branch_stream.getvalue()
                    emit_status_transition("python", "FAIL")
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

        emit_status_transition("python", "Pyright: type-check")
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
            emit_status_transition("python", "FAIL")
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

        emit_status_transition("python", pytest_step_name)
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
            emit_status_transition("python", "FAIL")
            return BranchResult(
                name="python",
                success=False,
                output=output,
                failed_step=pytest_step_name,
            )

        output = branch_stream.getvalue()
        emit_status_transition("python", "PASS")
        return BranchResult(name="python", success=True, output=output)

    def run_powershell_branch() -> BranchResult:
        branch_stream: StringIO = StringIO()
        branch_logger = StepLogger(stream=branch_stream)
        branch_runner = factory("powershell", branch_logger)

        emit_status_transition("powershell", "PoshQC: format")
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
            emit_status_transition("powershell", "FAIL")
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: format",
            )

        emit_status_transition("powershell", "PoshQC: analyze")
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
            emit_status_transition("powershell", "FAIL")
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: analyze",
            )

        emit_status_transition("powershell", "PoshQC: test")
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
            emit_status_transition("powershell", "FAIL")
            return BranchResult(
                name="powershell",
                success=False,
                output=output,
                failed_step="PoshQC: test",
            )

        output = branch_stream.getvalue()
        emit_status_transition("powershell", "PASS")
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
        result = func()
        results[name] = result
        if not result.success and not complete_all:
            cancel_event.set()

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

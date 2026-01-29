"""Task retry and verification helpers for atomic executor.

Purpose:
    Provides retry-with-backoff logic for Copilot invocations and
    QC verification helpers extracted from task_execution to maintain
    the 500-line file limit.

Flow:
    1. run_copilot_with_retry: Rate-limited, backoff-aware Copilot calls
    2. verify_task_qc: Post-Copilot QC verification with expect-fail support
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.dev_tools.atomic_executor.copilot_execution import log_msg, run_copilot
from scripts.dev_tools.atomic_executor.copilot_throttling import (
    CallRateLimiter,
    ExponentialBackoff,
    FailureKind,
    classify_copilot_failure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.dev_tools.atomic_executor.plan_parser import PlanTask


@dataclass
class RetryLoopResult:
    """Result of a throttle-aware Copilot retry loop.

    Attributes:
        success: True if Copilot completed successfully.
        exit_code: Final exit code (0 on success, 5 on failure).
        error_message: Human-readable error if failed, else None.
    """

    success: bool
    exit_code: int
    error_message: str | None = None


def run_copilot_with_retry(
    *,
    workspace: Path,
    prompt_text: str,
    log_file: Path,
    task_id: str,
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
    resume_session: bool,
    is_first_task: bool,
) -> RetryLoopResult:
    """
    Execute Copilot CLI with throttle-aware retry logic.

    Purpose:
        Handles rate limiting and exponential backoff for Copilot CLI calls,
        distinguishing throttle failures from non-recoverable errors.

    Args:
        workspace: Repository root path.
        prompt_text: Prompt to send to Copilot.
        log_file: Path to log file for recording events.
        task_id: Current task identifier for logging.
        preferred_model: Optional model name to force.
        run_id: Run identifier for artifact grouping.
        copilot_rate_limiter: Limiter for call frequency.
        copilot_backoff: Backoff strategy for throttle retries.
        copilot_max_retries: Maximum throttle retries allowed.
        copilot_output_tail_bytes: Bytes of output tail for failure classification.
        copilot_allow_shell: Allow shell commands without approval.
        copilot_allow_all_paths: Allow all file paths without approval.
        copilot_allow_all_urls: Allow all URLs without approval.
        copilot_trust_workspace: Persist workspace in trusted list.
        resume_session: Whether to resume an existing session.
        is_first_task: True if this is the first task in the run.

    Returns:
        RetryLoopResult: Success/failure status with optional error message.
    """
    copilot_invocation = 0
    throttle_retries = 0

    # Normalize negative max_retries to zero
    effective_max_retries = max(0, copilot_max_retries)

    while True:
        copilot_rate_limiter.acquire()

        copilot_result = run_copilot(
            workspace=workspace,
            prompt_text=prompt_text,
            log_file=log_file,
            task_id=task_id,
            preferred_model=preferred_model,
            run_id=run_id,
            resume_session=resume_session or copilot_invocation > 0,
            is_first_task=is_first_task,
            allow_all_paths=copilot_allow_all_paths,
            allow_all_urls=copilot_allow_all_urls,
            allow_shell=copilot_allow_shell,
            trust_workspace=copilot_trust_workspace,
            _output_tail_bytes=copilot_output_tail_bytes,
        )
        copilot_invocation += 1

        if copilot_result.exit_code == 0:
            # Success clears accumulated throttle state
            copilot_backoff.on_success()
            return RetryLoopResult(success=True, exit_code=0)

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
            log_msg(log_file, f"ERROR: {err_msg}")
            return RetryLoopResult(
                success=False,
                exit_code=5,
                error_message=err_msg,
            )

        # Throttle-like failure: bounded retry with backoff
        if throttle_retries >= effective_max_retries:
            err_msg = (
                f"Copilot CLI appears throttled, but max retries "
                f"({effective_max_retries}) were exhausted for task {task_id}."
            )
            print(err_msg, file=sys.stderr)
            log_msg(log_file, f"ERROR: {err_msg}")
            return RetryLoopResult(
                success=False,
                exit_code=5,
                error_message=err_msg,
            )

        delay_seconds = copilot_backoff.on_throttle()
        throttle_retries += 1

        retry_msg = (
            f"Copilot throttled for task {task_id}; retry "
            f"{throttle_retries}/{effective_max_retries} after "
            f"{delay_seconds:.2f}s backoff."
        )
        print(retry_msg)
        log_msg(log_file, f"WARN: {retry_msg}")

        # Apply backoff delay via injected sleeper for test determinism
        if delay_seconds > 0:
            copilot_rate_limiter.sleeper.sleep(delay_seconds)


@dataclass
class QCVerificationResult:
    """Result of post-Copilot QC verification.

    Attributes:
        passed: True if QC passed (or expected failure was achieved).
        should_retry: True if the task should be retried.
        retry_context: Context string for next retry attempt, if any.
        error_message: Human-readable error if failed, else None.
    """

    passed: bool
    should_retry: bool
    retry_context: str | None = None
    error_message: str | None = None


def verify_task_qc(
    task: PlanTask,
    qc_runner: object,
    log_file: Path,
    attempt: int,
) -> QCVerificationResult:
    """
    Run scoped QC verification after Copilot execution.

    Purpose:
        Handles the QC check-and-verify flow, including TDD Red (expect-fail)
        task semantics where pytest failure is the expected outcome.

    Args:
        task: The task being verified.
        qc_runner: QC runner instance with run_scoped() method.
        log_file: Path to log file for recording events.
        attempt: Current attempt number for retry context.

    Returns:
        QCVerificationResult: Verification status with retry guidance.

    Side Effects:
        Prints status messages and logs events.
    """
    try:
        # Use duck typing for qc_runner to support test doubles
        run_scoped = getattr(qc_runner, "run_scoped", None)
        if callable(run_scoped):
            run_scoped()

        # QC passed (no exception)
        if task.expect_fail:
            # Unexpected: test should have failed but all QC passed
            err_msg = f"Task {task.task_id} expected failure (TDD Red) but QC passed."
            print(err_msg, file=sys.stderr)
            log_msg(log_file, f"WARN: {err_msg}")

            return QCVerificationResult(
                passed=False,
                should_retry=True,
                retry_context=(
                    f"Attempt {attempt}: Expected pytest failure but all QC passed.\n"
                    "The test should fail to verify the TDD Red condition."
                ),
            )

        # Normal task: QC passed
        return QCVerificationResult(passed=True, should_retry=False)

    except subprocess.CalledProcessError as e:
        # Determine which command failed to distinguish pytest from other tools
        if isinstance(e.cmd, str):
            cmd_str = e.cmd
        else:
            cmd_str = " ".join(str(arg) for arg in e.cmd)
        is_pytest_failure = "pytest" in cmd_str

        if task.expect_fail and is_pytest_failure:
            # SUCCESS: Expected pytest failure achieved (TDD Red workflow)
            success_msg = f"Task {task.task_id} failed as expected (TDD Red). Verified."
            print(success_msg)
            log_msg(log_file, f"SUCCESS: {success_msg}")
            return QCVerificationResult(passed=True, should_retry=False)

        # Real failure (non-pytest error OR normal task with any failure)
        err_msg = f"Scoped QC failed for task {task.task_id}: {e}"
        print(err_msg, file=sys.stderr)
        log_msg(log_file, f"WARN: {err_msg}")

        return QCVerificationResult(
            passed=False,
            should_retry=True,
            retry_context=(
                f"Attempt {attempt} failed verification.\n"
                f"Error: {e}\n"
                "Please fix code/test issues and try again."
            ),
            error_message=err_msg,
        )

"""QC orchestration helpers for atomic executor.

Purpose:
    Handles preflight QC, auto-QC phases, and QC fix loops.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.dev_tools.atomic_executor.cli import LOG_DIR
from scripts.dev_tools.atomic_executor.copilot_execution import log_msg, run_copilot
from scripts.dev_tools.atomic_executor.copilot_throttling import (
    CallRateLimiter,
    ExponentialBackoff,
    FailureKind,
    classify_copilot_failure,
)
from scripts.dev_tools.atomic_executor.plan_parser import (
    AutoQCPhase,
    PlanParser,
)
from scripts.dev_tools.atomic_executor.pytest_expectations import (
    ResolvedTestExpectations,
    parse_pytest_failure_output,
    resolve_checked_test_expectations,
)
from scripts.dev_tools.atomic_executor.qc_runner import QCLoopResult, QCRunner


def build_qc_fix_prompt(
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


@dataclass(frozen=True)
class PreflightQCResult:
    """
    Result of a pre-flight QC run with captured output.

    Attributes:
        success: True if all QC steps passed.
        output: Combined stdout/stderr from all QC steps.
        failed_step: Name of the first step that failed (or None if success).
    """

    success: bool
    output: str
    failed_step: str | None = None


def resolve_plan_expectations(
    parser: PlanParser,
) -> ResolvedTestExpectations | None:
    """
    Resolve checked plan expectations for preflight gating.

    Purpose:
        Determine if the active plan includes checked expectation tasks.

    Args:
        parser (PlanParser): Plan parser for the current plan file.

    Returns:
        ResolvedTestExpectations | None: Resolved expectations, or None when empty.
    """
    plan = parser.parse()
    expectations = resolve_checked_test_expectations(plan)
    if (
        not expectations.expected_fail_refs
        and not expectations.expected_pass_refs
        and not expectations.missing_test_refs
    ):
        return None
    return expectations


def matches_expected_ref(nodeid: str, expected_refs: set[str]) -> bool:
    """
    Check whether a failing nodeid matches any expected ref prefix.

    Purpose:
        Support prefix matching to handle parameterized pytest nodeids.

    Args:
        nodeid (str): Failing pytest nodeid.
        expected_refs (set[str]): Expected nodeid prefixes to match against.

    Returns:
        bool: True when a prefix match is found.
    """
    # Scan the expected refs to allow prefix matching for parametrized tests.
    for expected_ref in expected_refs:
        if nodeid.startswith(expected_ref):
            return True
    return False


def run_preflight_qc_with_capture(
    workspace: Path,
    *,
    expectations: ResolvedTestExpectations | None = None,
) -> PreflightQCResult:
    """
    Run full QC toolchain and capture combined output.

    Purpose:
        Execute Black/Ruff/Pyright/Pytest in order and capture all output
        for use in the pre-flight fix prompt.

    Args:
        workspace: Repository root.

    Returns:
        PreflightQCResult: Success status and captured output.
    """
    steps = [
        ("black", ["poetry", "run", "black", "--check", "."]),
        ("ruff", ["poetry", "run", "ruff", "check"]),
        ("pyright", ["poetry", "run", "pyright"]),
        (
            "pytest",
            [
                "poetry",
                "run",
                "pytest",
                "--color=no",
                "--cov=src/lexile_corpus_tuner",
                "--cov=scripts/dev_tools",
                "--cov-report=term-missing",
            ],
        ),
    ]

    all_output: list[str] = []
    expected_refs: set[str] = set()

    if expectations is not None:
        expected_refs = (
            expectations.expected_fail_refs | expectations.expected_pass_refs
        )
        if expectations.missing_test_refs:
            missing_refs = ", ".join(expectations.missing_test_refs)
            message = (
                "Missing test reference for expectation-tagged tasks: "
                f"{missing_refs}"
            )
            return PreflightQCResult(
                success=False,
                output=message,
                failed_step="pytest-collect",
            )

    for step_name, cmd in steps:
        all_output.append(f"=== {step_name.upper()} ===")

        if step_name == "pytest" and expectations is not None and expected_refs:
            all_output.append("=== PYTEST COLLECT ===")
            collect_cmd = [
                "poetry",
                "run",
                "pytest",
                "--collect-only",
                "--color=no",
                *sorted(expected_refs),
            ]
            collect_result = subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
                collect_cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            collect_output = (collect_result.stdout or "") + (
                collect_result.stderr or ""
            )
            all_output.append(
                collect_output.strip() if collect_output else "(no output)"
            )
            if collect_result.returncode != 0:
                return PreflightQCResult(
                    success=False,
                    output="\n\n".join(all_output),
                    failed_step="pytest-collect",
                )

        # Commands are static hardcoded constants (poetry run black/ruff/pyright/pytest)
        result = subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        all_output.append(combined.strip() if combined else "(no output)")

        if result.returncode != 0:
            if step_name != "pytest" or expectations is None:
                return PreflightQCResult(
                    success=False,
                    output="\n\n".join(all_output),
                    failed_step=step_name,
                )

            summary = parse_pytest_failure_output(combined)
            if summary.has_collection_error:
                all_output.append(
                    "Pytest collection/import errors detected; failing QC."
                )
                return PreflightQCResult(
                    success=False,
                    output="\n\n".join(all_output),
                    failed_step=step_name,
                )

            unexpected_failures: list[str] = []
            expected_pass_hits: list[str] = []

            # Compare failing nodeids against expected refs with prefix matching.
            for nodeid in summary.failed_nodeids:
                if matches_expected_ref(nodeid, expectations.expected_pass_refs):
                    expected_pass_hits.append(nodeid)
                    unexpected_failures.append(nodeid)
                elif matches_expected_ref(nodeid, expectations.expected_fail_refs):
                    continue
                else:
                    unexpected_failures.append(nodeid)

            if unexpected_failures:
                all_output.append("Unexpected pytest failures detected.")
                if expected_pass_hits:
                    all_output.append(
                        "Expected-pass override applied to: "
                        + ", ".join(expected_pass_hits)
                    )
                return PreflightQCResult(
                    success=False,
                    output="\n\n".join(all_output),
                    failed_step=step_name,
                )

            all_output.append(
                "Expected pytest failures allowed: "
                + ", ".join(sorted(summary.failed_nodeids))
            )

    return PreflightQCResult(
        success=True,
        output="\n\n".join(all_output),
    )


def build_preflight_qc_fix_prompt(workspace: Path, qc_output: str) -> str:
    """
    Build a prompt directing Copilot to fix pre-flight QC failures.

    Purpose:
        Creates a focused prompt that instructs the LLM to fix baseline QC
        issues, run the full toolchain itself, and iterate until all checks
        pass before yielding control back to the executor.

    Args:
        workspace: Repository root path for context.
        qc_output: Captured output from the failed QC run.

    Returns:
        str: Prompt text for Copilot CLI execution.
    """
    return (
        "# Pre-flight QC Fix Required\n\n"
        "The atomic executor detected baseline QC failures before task execution.\n"
        "You must fix these issues before the plan can proceed.\n\n"
        f"**Workspace:** `{workspace.as_posix()}`\n\n"
        "## Failed QC Output\n\n"
        "```\n"
        f"{qc_output}\n"
        "```\n\n"
        "## Your Instructions\n\n"
        "1. Analyze the QC failures above.\n"
        "2. Make the minimal code changes required to fix each issue.\n"
        "3. **Run the full QC toolchain yourself** to verify your fixes:\n"
        "   - `poetry run black .`\n"
        "   - `poetry run ruff check`\n"
        "   - `poetry run pyright`\n"
        "   - `poetry run pytest --cov=src/lexile_corpus_tuner "
        "--cov=scripts/dev_tools --cov-report=term-missing`\n"
        "4. If any step fails, fix the issues and re-run from step 3.\n"
        "5. **Do NOT end your turn until all QC steps pass.**\n"
        "6. Once all checks pass, reply with a brief summary of what you fixed.\n\n"
        "**CRITICAL:** You must iterate until QC passes completely. "
        "The executor will verify QC independently after you yield control.\n"
    )


def run_preflight_qc_fix_loop(
    *,
    workspace: Path,
    log_file: Path,
    run_id: str,
    preferred_model: str | None,
    copilot_rate_limiter: CallRateLimiter,
    copilot_backoff: ExponentialBackoff,
    copilot_max_retries: int,
    copilot_output_tail_bytes: int,
    copilot_allow_shell: bool,
    copilot_allow_all_paths: bool,
    copilot_allow_all_urls: bool,
    copilot_trust_workspace: bool,
    max_fix_attempts: int,
    expectations: ResolvedTestExpectations | None,
) -> int:
    """
    Run the pre-flight QC fix loop until baseline passes or attempts exhausted.

    Purpose:
        When pre-flight QC fails, this function invokes Copilot to fix the
        issues. The LLM is instructed to run the toolchain itself and iterate
        until passing. After Copilot yields, the executor verifies QC
        independently. If still failing, the loop retries.

    Flow:
        1. Build prompt with QC failure output
        2. Invoke Copilot (LLM runs toolchain itself)
        3. When Copilot returns, run full QC to verify
        4. If QC passes, return 0
        5. If QC fails, retry (up to max_fix_attempts)
        6. If attempts exhausted, return error code

    Args:
        workspace: Repository root.
        log_file: Path to log file.
        run_id: Run identifier for artifact grouping.
        preferred_model: Preferred AI model name.
        copilot_rate_limiter: Rate limiter for Copilot calls.
        copilot_backoff: Backoff strategy for throttling.
        copilot_max_retries: Max throttle retries per invocation.
        copilot_output_tail_bytes: Bytes of output tail to retain.
        copilot_allow_shell: Allow shell commands without approval.
        copilot_allow_all_paths: Allow all file paths without approval.
        copilot_allow_all_urls: Allow all URLs without approval.
        copilot_trust_workspace: Add workspace to trusted folders.
        max_fix_attempts: Max number of fix attempts (0 = infinite).
        expectations: Optional resolved plan expectations for pytest gating.

    Returns:
        int: 0 on success, 6 on failure.
    """
    attempt = 1
    copilot_invocation_count = 0

    while True:
        if max_fix_attempts > 0 and attempt > max_fix_attempts:
            msg = f"Pre-flight QC fix failed after {max_fix_attempts} attempts."
            print(msg, file=sys.stderr)
            log_msg(log_file, f"ERROR: {msg}")
            return 6

        # Capture QC output for the prompt
        print("Running pre-flight QC check...")
        log_msg(log_file, "INFO: Running pre-flight QC check")

        preflight_check = run_preflight_qc_with_capture(
            workspace, expectations=expectations
        )
        if preflight_check.success:
            # QC passed - we're done
            print("Pre-flight QC passed.")
            log_msg(log_file, "INFO: Pre-flight QC passed")
            return 0

        # QC failed - extract output for prompt
        qc_output = preflight_check.output

        limit_str = str(max_fix_attempts) if max_fix_attempts > 0 else "∞"
        msg = (
            f"Pre-flight QC failed (attempt {attempt}/{limit_str}), "
            "invoking Copilot to fix..."
        )
        print(msg)
        log_msg(log_file, f"WARN: {msg}")

        # Build prompt for Copilot
        prompt_text = build_preflight_qc_fix_prompt(workspace, qc_output)

        # Write prompt to file for debugging
        prompt_dir = workspace / LOG_DIR / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / f"prompt_{run_id}_preflight_{attempt}.md"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        # Invoke Copilot with throttle-aware loop
        throttle_retries = 0

        while True:
            copilot_rate_limiter.acquire()

            copilot_result = run_copilot(
                workspace=workspace,
                prompt_text=prompt_text,
                log_file=log_file,
                task_id=f"preflight-{attempt}",
                preferred_model=preferred_model,
                run_id=run_id,
                resume_session=(copilot_invocation_count > 0),
                is_first_task=(copilot_invocation_count == 0),
                allow_all_paths=copilot_allow_all_paths,
                allow_all_urls=copilot_allow_all_urls,
                allow_shell=copilot_allow_shell,
                trust_workspace=copilot_trust_workspace,
                _output_tail_bytes=copilot_output_tail_bytes,
            )
            copilot_invocation_count += 1

            if copilot_result.exit_code == 0:
                copilot_backoff.on_success()
                break

            failure_kind = classify_copilot_failure(
                exit_code=copilot_result.exit_code,
                output_tail=copilot_result.output_tail,
            )
            if failure_kind is FailureKind.NON_THROTTLE:
                err_msg = (
                    "Copilot CLI failed (non-throttle) during pre-flight fix. "
                    f"exit_code={copilot_result.exit_code}. "
                    f"output_tail={copilot_result.output_tail!r}"
                )
                print(err_msg, file=sys.stderr)
                log_msg(log_file, f"ERROR: {err_msg}")
                return 6

            # Throttle handling
            effective_max_retries = copilot_max_retries
            if effective_max_retries < 0:
                effective_max_retries = 0

            if throttle_retries >= effective_max_retries:
                err_msg = (
                    f"Copilot CLI throttled during pre-flight fix, "
                    f"max retries ({effective_max_retries}) exhausted."
                )
                print(err_msg, file=sys.stderr)
                log_msg(log_file, f"ERROR: {err_msg}")
                return 6

            delay_seconds = copilot_backoff.on_throttle()
            throttle_retries += 1

            retry_msg = (
                f"Copilot throttled during pre-flight fix; retry "
                f"{throttle_retries}/{effective_max_retries} after "
                f"{delay_seconds:.2f}s backoff."
            )
            print(retry_msg)
            log_msg(log_file, f"WARN: {retry_msg}")

            if delay_seconds > 0:
                copilot_rate_limiter.sleeper.sleep(delay_seconds)

        # Copilot returned - verify QC independently
        print("Copilot completed, verifying QC...")
        log_msg(log_file, "INFO: Verifying QC after Copilot pre-flight fix")
        attempt += 1
        # Loop back to re-run QC check at the top


def execute_auto_qc_phase(
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
            log_msg(log_file, f"ERROR: {msg}")
            print(f"See log: {log_file}", file=sys.stderr)
            return 5

        try:
            result = qc_runner.run_full_loop_with_artifacts(
                artifact_paths=phase.artifact_paths,
            )
        except RuntimeError as exc:
            err_msg = f"Auto-QC phase {phase.phase} failed: {exc}"
            print(err_msg, file=sys.stderr)
            log_msg(log_file, f"ERROR: {err_msg}")
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
            log_msg(log_file, f"SUCCESS: {success_msg}")
            return 0

        # If the loop failed but we lack detail, stop with actionable logs.
        if result.failure is None:
            err_msg = f"Auto-QC phase {phase.phase} failed without error details."
            print(err_msg, file=sys.stderr)
            log_msg(log_file, f"ERROR: {err_msg}")
            return 5

        # Build fix-only prompt and invoke Copilot when QC fails.
        prompt_text = build_qc_fix_prompt(
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
                log_msg(log_file, f"ERROR: {err_msg}")
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
                log_msg(log_file, f"ERROR: {err_msg}")
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
            log_msg(log_file, f"WARN: {retry_msg}")

            # Apply backoff delay via injected sleeper for deterministic tests.
            if delay_seconds > 0:
                copilot_rate_limiter.sleeper.sleep(delay_seconds)

        attempt += 1

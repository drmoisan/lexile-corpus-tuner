"""Task execution and main entry point for atomic executor.

Purpose:
    Handles single-task execution with retries and main CLI entry point.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev_tools.atomic_executor.arg_parser import parse_args
from scripts.dev_tools.atomic_executor.cli import (
    LOG_DIR,
    first_non_read_task,
    handle_shutdown_signal,
    is_phase0_read_task,
    is_shutdown_requested,
    phase0_read_tasks,
)
from scripts.dev_tools.atomic_executor.clipboard_helpers import copy_to_clipboard
from scripts.dev_tools.atomic_executor.copilot_execution import log_msg
from scripts.dev_tools.atomic_executor.copilot_throttling import (
    CallRateLimiter,
    ExponentialBackoff,
    SystemClock,
    SystemRandom,
    TimeSleeper,
)
from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver
from scripts.dev_tools.atomic_executor.plan_discovery import resolve_feature_plan
from scripts.dev_tools.atomic_executor.plan_parser import (
    AutoQCPhase,
    PlanParser,
    PlanTask,
)
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_orchestration import (
    execute_auto_qc_phase,
    resolve_plan_expectations,
    run_preflight_qc_fix_loop,
)
from scripts.dev_tools.atomic_executor.qc_runner import QCRunner
from scripts.dev_tools.atomic_executor.task_retry import (
    run_copilot_with_retry,
    verify_task_qc,
)
from scripts.dev_tools.atomic_executor.workspace_helpers import (
    acquire_executor_lock,
    refuse_protected_branch,
    release_executor_lock,
    resolve_workspace,
)


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
    """Execute a single atomic task with retries and QC verification.

    Returns 0 on success, 5 on failure after max attempts.
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
        return execute_auto_qc_phase(
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
            log_msg(log_file, f"ERROR: {msg}")
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
        log_msg(log_file, f"INFO: {msg}")

        # Throttle-aware Copilot invocation via extracted helper
        copilot_result = run_copilot_with_retry(
            workspace=workspace,
            prompt_text=prompt_text,
            log_file=log_file,
            task_id=cur.task_id,
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
            resume_session=(attempt > 1),
            is_first_task=is_first_task,
        )

        if not copilot_result.success:
            print(f"See log: {log_file}", file=sys.stderr)
            return copilot_result.exit_code

        # Refresh plan/task state after Copilot run
        cur_after = parser.find_task_by_id(cur.task_id)

        # Task-step QC verification via extracted helper
        qc_result = verify_task_qc(
            task=cur,
            qc_runner=qc_runner,
            log_file=log_file,
            attempt=attempt,
        )

        if qc_result.passed:
            # For expect-fail tasks, checkbox was handled in verify_task_qc
            if cur.expect_fail:
                if cur_after and not cur_after.checked:
                    parser.flip_checkbox(cur_after)
                return 0

            # Normal task success: flip checkbox if model didn't
            if cur_after and not cur_after.checked:
                parser.flip_checkbox(cur_after)

            success_msg = f"Task {cur.task_id} complete and gated."
            print(success_msg)
            log_msg(log_file, f"SUCCESS: {success_msg}")
            return 0

        # QC failed - retry if allowed
        if qc_result.should_retry:
            retry_ctx = qc_result.retry_context
            attempt += 1
            continue

        # Unexpected state: should not reach here
        return 5


def main(argv: list[str] | None = None) -> int:
    """Main entry point for atomic executor CLI.

    Orchestrates feature resolution, plan parsing, QC, and Copilot invocation.
    Returns 0 on success, non-zero on error.
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
    # Declare global for signal handler to access; set below if execute-all
    global _active_lock_path
    if args.cmd == "execute-all":
        lock_path = acquire_executor_lock(workspace)
        _active_lock_path = lock_path

    # Register signal handlers for graceful shutdown (Ctrl+C, kill)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        # Parse plan and preflight validate
        parser = PlanParser(plan_path)
        parser.preflight_validate()

        # Determine current task
        if args.cmd == "resume" or args.cmd == "execute-all":
            cur = parser.next_unchecked_task()
            if cur is None:
                print("Plan already complete: no unchecked tasks found.")
                return 0
        elif args.start:
            cur = parser.find_task_by_id(args.start)
        else:
            cur = parser.next_unchecked_task()
            if cur is None:
                print("Plan already complete: no unchecked tasks found.")
                return 0

        include_phase0_reads = False
        phase0_reads = phase0_read_tasks(parser)
        if phase0_reads:
            include_phase0_reads = True

        # Bundle Phase 0 read tasks with the first non-read task on session start.
        if include_phase0_reads and is_phase0_read_task(cur):
            non_read_task = first_non_read_task(parser)
            if non_read_task is not None:
                cur = non_read_task

        builder = PromptBuilder(
            workspace,
            prompt_template_path,
            preferred_model=args.preferred_model,
        )
        qc_runner = QCRunner(workspace)
        preflight_expectations = resolve_plan_expectations(parser)

        # Per-run throttling controls. The limiter must persist across tasks to
        # regulate overall call cadence.
        copilot_rate_limiter = CallRateLimiter(
            max_calls=args.copilot_cli_max_calls_per_window,
            window_seconds=args.copilot_cli_window_seconds,
            clock=SystemClock(),
            sleeper=TimeSleeper(),
        )

        # Pre-flight QC: run full toolchain before task execution
        # If baseline fails, enter fix loop with Copilot
        if not args.skip_preflight_qc:
            # Backoff state for pre-flight Copilot invocations
            preflight_backoff = ExponentialBackoff(
                base_seconds=args.copilot_cli_backoff_base_seconds,
                max_seconds=args.copilot_cli_backoff_max_seconds,
                random_source=SystemRandom(),
            )
            preflight_result = run_preflight_qc_fix_loop(
                workspace=workspace,
                log_file=log_file,
                run_id=run_id,
                preferred_model=args.preferred_model,
                copilot_rate_limiter=copilot_rate_limiter,
                copilot_backoff=preflight_backoff,
                copilot_max_retries=args.copilot_cli_max_retries,
                copilot_output_tail_bytes=args.copilot_cli_output_tail_bytes,
                copilot_allow_shell=args.copilot_allow_shell,
                copilot_allow_all_paths=args.copilot_allow_all_paths,
                copilot_allow_all_urls=args.copilot_allow_all_urls,
                copilot_trust_workspace=args.copilot_trust_workspace,
                max_fix_attempts=args.max_fix_attempts,
                expectations=preflight_expectations,
            )
            if preflight_result != 0:
                return preflight_result

        is_first_task = True

        while True:
            # Check for graceful shutdown request (Ctrl+C or SIGTERM)
            if is_shutdown_requested():
                print("[atomic_executor] Shutdown requested, exiting after cleanup.")
                return 130  # Standard exit code for SIGINT

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
                        phase_expectations = resolve_plan_expectations(parser)
                        qc_runner.run_full(expectations=phase_expectations)
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
        # Clear global and release lock
        _active_lock_path = None
        if lock_path is not None:
            release_executor_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

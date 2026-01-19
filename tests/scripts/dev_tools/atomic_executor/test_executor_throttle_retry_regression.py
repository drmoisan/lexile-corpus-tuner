"""Regression tests for executor behavior under simulated Copilot throttling.

These tests intentionally simulate Copilot CLI failures without invoking the real
`copilot` binary and without performing any filesystem writes.

The regression covered here matches issue #80: a throttling-style Copilot failure
should not abort the run; instead, the executor should retry the same task and
preserve task ordering until the retry succeeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures

from scripts.dev_tools.atomic_executor.copilot_runner import CopilotRunResult
from scripts.dev_tools.atomic_executor.copilot_throttling import (
    CallRateLimiter,
    ExponentialBackoff,
)
from scripts.dev_tools.atomic_executor.plan_parser import PlanParser, PlanTask
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder
from scripts.dev_tools.atomic_executor.qc_runner import QCRunner


class _FakePromptBuilder:
    """Deterministic prompt builder stub.

    Purpose:
        Avoid filesystem/template dependencies while exercising executor control
        flow.

    Notes:
        This stub returns a constant prompt string. The prompt contents are
        irrelevant to the retry behavior under test.
    """

    def build(
        self,
        _feature_dir: Path,
        _task: PlanTask,
        retry_context: str | None = None,
        include_phase0_reads: bool = False,
    ) -> str:
        """Return a stable prompt string.

        Args:
            _feature_dir (Path): Unused.
            _task (PlanTask): Unused.
            retry_context (str | None): Unused.
            include_phase0_reads (bool): Unused flag for phase 0 read bundling.

        Returns:
            str: Constant prompt text.
        """

        _ = include_phase0_reads
        return "PROMPT"


class _FakePlanParser:
    """In-memory plan parser stub.

    Purpose:
        Avoid disk access while providing the minimal PlanParser API surface
        used by `_execute_one_task()`.
    """

    def __init__(self, task: PlanTask) -> None:
        self._task = task

    def find_task_by_id(self, task_id: str) -> PlanTask:
        """Return the known task by id.

        Args:
            task_id (str): Task identifier.

        Returns:
            PlanTask: The stored task.

        Raises:
            RuntimeError: If the id does not match the stored task.
        """

        if task_id != self._task.task_id:
            raise RuntimeError(f"Unexpected task id: {task_id}")
        return self._task

    def flip_checkbox(self, _task_to_check: PlanTask) -> None:
        """Fail if called.

        Purpose:
            The regression scenario asserts ordering (no advancement) but does
            not need checkbox mutation. This stub intentionally fails if called
            to ensure the test does not perform filesystem writes.

        Raises:
            AssertionError: Always.
        """

        raise AssertionError("flip_checkbox() should not be invoked in this test")


class _FakeQCRunner:
    """Stub QC runner.

    Purpose:
        The regression focuses on Copilot retry behavior. Scoped QC should run
        only after a successful Copilot invocation.
    """

    def __init__(self) -> None:
        self.scoped_runs: int = 0

    def run_scoped(self) -> None:
        """Record a scoped QC run."""

        self.scoped_runs += 1


class _StaticClock:
    """Clock stub returning a constant monotonic time."""

    def now_monotonic(self) -> float:
        """Return a fixed monotonic timestamp."""

        return 0.0


class _NoOpSleeper:
    """Sleeper stub that does not actually sleep."""

    def sleep(self, seconds: float) -> None:
        """No-op sleep used to keep tests deterministic."""

        _ = seconds


def _noop_log(_log_file: Path, _msg: str) -> None:
    """Ignore log writes from the CLI under test.

    Purpose:
        The CLI implementation writes progress logs to disk. Unit tests in this
        repo must not write temporary files, so we patch the logger with a typed
        no-op.
    """


class _ZeroRandom:
    """Random source that always returns 0.0 (full jitter -> 0 delay)."""

    def random(self) -> float:
        """Return 0.0."""

        return 0.0


def test_execute_one_task_retries_on_throttle_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Throttle->success should retry within the same task.

    Scenario:
        Simulate a Copilot outcome sequence: throttle failure on first attempt,
        then success on the second attempt.

    Expected:
        The executor retries (calls Copilot twice) and succeeds without
        attempting to advance to the next task.

    Note:
        This regression originally failed before issue #80 was implemented.
    """

    from scripts.dev_tools.atomic_executor import cli as cli_mod

    # Arrange: prevent all filesystem writes from the code under test.
    monkeypatch.setattr(cli_mod, "_log_msg", _noop_log)

    # Arrange: create a task that is already checked so checkbox-flipping is not needed.
    task = PlanTask(
        task_id="P1-T1",
        phase=1,
        task_num=1,
        title="regression",
        checked=True,
        line_index=0,
    )

    fake_parser = _FakePlanParser(task)
    fake_builder = _FakePromptBuilder()
    fake_qc = _FakeQCRunner()

    calls: list[str] = []

    def _fake_run_copilot(**_kwargs: object) -> CopilotRunResult:
        """Simulate throttle failure once, then succeed.

        Purpose:
            The executor should treat an initial throttle-like failure as
            retryable, so it should invoke Copilot again for the same task.
            This behavior is implemented later in the plan.
        """

        calls.append("copilot")
        if len(calls) == 1:
            return CopilotRunResult(
                exit_code=1,
                output_tail="Error: rate limit exceeded (HTTP 429)",
            )
        return CopilotRunResult(exit_code=0, output_tail="")

    monkeypatch.setattr(cli_mod, "run_copilot", _fake_run_copilot)

    # Arrange: throttling policy inputs (fully deterministic; no real sleeps).
    copilot_rate_limiter = CallRateLimiter(
        max_calls=100,
        window_seconds=60.0,
        clock=_StaticClock(),
        sleeper=_NoOpSleeper(),
    )
    copilot_backoff = ExponentialBackoff(
        base_seconds=1.0,
        max_seconds=1.0,
        random_source=_ZeroRandom(),
    )

    # Act: attempt to execute the task.
    exit_code = cli_mod.execute_one_task(
        workspace=Path("/repo"),  # noqa: S108 - test fixture path
        cur=task,
        parser=cast(PlanParser, fake_parser),
        builder=cast(PromptBuilder, fake_builder),
        qc_runner=cast(QCRunner, fake_qc),
        log_file=Path("/dev/null"),  # noqa: S108 - test fixture path
        prompt_template_path=Path("/template.md"),  # noqa: S108 - test fixture path
        max_fix_attempts=3,
        feature_dir=Path("/feature"),  # noqa: S108 - test fixture path
        preferred_model=None,
        run_id="run",
        copilot_rate_limiter=copilot_rate_limiter,
        copilot_backoff=copilot_backoff,
        copilot_max_retries=2,
        copilot_output_tail_bytes=4096,
        copilot_allow_shell=True,
        copilot_allow_all_paths=True,
        copilot_allow_all_urls=False,
        copilot_trust_workspace=True,
        include_phase0_reads=False,
    )

    # Assert: success and expected sequencing.
    assert exit_code == 0
    assert calls == ["copilot", "copilot"]
    assert fake_qc.scoped_runs == 1

"""Unit test for bounded throttle retries in the atomic executor.

This verifies an important requirement for issue #80:
- Throttle-triggered retry behavior is bounded by configuration (no infinite
  loops by default). When Copilot remains throttled, the executor must terminate
  with a non-zero exit code after exhausting max retries.

Design constraints:
- Deterministic: no real sleeps/time/subprocess/Copilot.
- No filesystem writes: patch logging and use in-memory fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _noop_log(_log_file: Path, _msg: str) -> None:
    """Ignore log writes from the CLI under test.

    Purpose:
        Unit tests must avoid filesystem writes. The executor logs progress to a
        file, so we patch the logger with a typed no-op.
    """


class _FakeClock:
    """Deterministic monotonic clock for tests."""

    def __init__(self, start: float) -> None:
        self._now = start

    def now_monotonic(self) -> float:
        """Return the current monotonic time."""

        return self._now


class _FakeSleeper:
    """Sleeper that records delays (no real sleeping)."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        """Record the sleep duration."""

        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self.sleeps.append(seconds)


class _FakeRandom:
    """Deterministic RandomSource for jitter."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        """Return a fixed jitter value."""

        return self._value


@dataclass
class _FakePlanParser:
    """Minimal PlanParser fake for bounded retry tests."""

    task_after: PlanTask
    flip_called: bool = False

    def find_task_by_id(self, task_id: str) -> PlanTask:
        """Return a fixed task model."""

        _ = task_id
        return self.task_after

    def flip_checkbox(self, task_to_check: PlanTask) -> None:
        """Record that a checkbox flip was attempted."""

        _ = task_to_check
        self.flip_called = True


@dataclass
class _FakePromptBuilder:
    """Minimal PromptBuilder fake."""

    prompt_text: str

    def build(
        self,
        feature_dir: Path,
        cur: PlanTask,
        retry_context: str | None = None,
        include_phase0_reads: bool = False,
    ) -> str:
        """Return a stable prompt string.

        Purpose:
            Provide a deterministic prompt for executor tests without touching
            the filesystem or relying on templates.

        Args:
            feature_dir (Path): Unused path to the feature directory.
            cur (PlanTask): Unused task metadata.
            retry_context (str | None): Unused retry context.
            include_phase0_reads (bool): Unused flag for phase 0 read bundling.

        Returns:
            str: Constant prompt text.
        """

        _ = feature_dir
        _ = cur
        _ = retry_context
        _ = include_phase0_reads
        return self.prompt_text


@dataclass
class _FakeQCRunner:
    """Minimal QCRunner fake."""

    scoped_calls: int = 0

    def run_scoped(self) -> None:
        """Record that scoped QC ran."""

        self.scoped_calls += 1


def test_executor_terminates_after_max_throttle_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-throttled Copilot calls must stop after max retries."""

    from scripts.dev_tools.atomic_executor import cli

    # Patch logging helper to avoid writing log files during tests.
    monkeypatch.setattr(cli, "_log_msg", _noop_log)

    copilot_calls: list[int] = []

    def _fake_run_copilot(**_kwargs: object) -> CopilotRunResult:
        """Always return a throttle-like failure without invoking subprocess."""

        copilot_calls.append(1)
        return CopilotRunResult(exit_code=1, output_tail="429 too many requests")

    monkeypatch.setattr(cli, "run_copilot", _fake_run_copilot)

    cur = PlanTask(
        task_id="P3-T2",
        phase=3,
        task_num=2,
        title="Add retry loop",
        checked=False,
        line_index=0,
    )

    task_after = PlanTask(
        task_id=cur.task_id,
        phase=cur.phase,
        task_num=cur.task_num,
        title=cur.title,
        checked=False,
        line_index=0,
    )

    parser = _FakePlanParser(task_after=task_after)
    builder = _FakePromptBuilder(prompt_text="prompt")
    qc_runner = _FakeQCRunner()

    clock = _FakeClock(start=100.0)
    sleeper = _FakeSleeper()
    limiter = CallRateLimiter(
        max_calls=100, window_seconds=60.0, clock=clock, sleeper=sleeper
    )

    backoff = ExponentialBackoff(
        base_seconds=1.0,
        max_seconds=1.0,
        random_source=_FakeRandom(1.0),
    )

    max_retries = 2
    exit_code = cli.execute_one_task(
        workspace=Path("repo"),
        cur=cur,
        parser=cast(PlanParser, parser),
        builder=cast(PromptBuilder, builder),
        qc_runner=cast(QCRunner, qc_runner),
        log_file=Path("log.txt"),
        prompt_template_path=Path("prompt_template.md"),
        max_fix_attempts=1,
        feature_dir=Path("feature"),
        preferred_model=None,
        run_id="run",
        copilot_rate_limiter=limiter,
        copilot_backoff=backoff,
        copilot_max_retries=max_retries,
        copilot_output_tail_bytes=256,
        copilot_allow_shell=True,
        copilot_allow_all_paths=True,
        copilot_allow_all_urls=False,
        copilot_trust_workspace=True,
        include_phase0_reads=False,
        print_prompt=False,
        copy_prompt=False,
    )

    assert exit_code != 0

    # With max_retries=2: attempt 1 throttled -> retry 1 (sleep)
    # attempt 2 throttled -> retry 2 (sleep)
    # attempt 3 throttled -> retries exhausted -> fail
    assert len(copilot_calls) == 3
    assert sleeper.sleeps == [1.0, 1.0]

    # Executor must not run QC or flip checkboxes when Copilot never succeeds.
    assert qc_runner.scoped_calls == 0
    assert parser.flip_called is False

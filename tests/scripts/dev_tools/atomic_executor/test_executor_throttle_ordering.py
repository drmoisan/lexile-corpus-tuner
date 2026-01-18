"""Unit test for executor ordering under Copilot CLI throttling.

This verifies an important invariant for issue #80:
- When a Copilot invocation is throttled, the executor retries within the same
  task and does *not* advance the plan (checkbox flip / task completion) until a
  Copilot invocation succeeds.

Design constraints:
- Deterministic: no real sleeps, no real time, no subprocess, no Copilot CLI.
- No filesystem writes: we patch logging and avoid PlanParser disk operations.
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

    def __init__(self, call_log: list[str]) -> None:
        self.sleeps: list[float] = []
        self._call_log = call_log

    def sleep(self, seconds: float) -> None:
        """Record the sleep duration."""

        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self.sleeps.append(seconds)
        self._call_log.append(f"sleep:{seconds}")


class _FakeRandom:
    """Deterministic RandomSource for jitter."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        """Return a fixed jitter value."""

        return self._value


@dataclass
class _FakePlanParser:
    """Minimal PlanParser fake.

    Purpose:
        Avoid filesystem writes in tests while still allowing `_execute_one_task`
        to query task state and attempt a checkbox flip.

    Attributes:
        task_after (PlanTask): The task returned by `find_task_by_id`.
        call_log (list[str]): Shared call log for ordering assertions.
    """

    task_after: PlanTask
    call_log: list[str]

    def find_task_by_id(self, task_id: str) -> PlanTask:
        """Return a fixed task model (simulates reading updated plan state)."""

        self.call_log.append(f"find:{task_id}")
        return self.task_after

    def flip_checkbox(self, task_to_check: PlanTask) -> None:
        """Record a checkbox flip attempt (no disk I/O)."""

        self.call_log.append(f"flip:{task_to_check.task_id}")


@dataclass
class _FakePromptBuilder:
    """Minimal PromptBuilder fake.

    Purpose:
        Produce a stable prompt without reading feature folder files.
    """

    prompt_text: str

    def build(
        self, feature_dir: Path, cur: PlanTask, retry_context: str | None = None
    ) -> str:
        """Return a stable prompt string."""

        _ = feature_dir
        _ = cur
        _ = retry_context
        return self.prompt_text


@dataclass
class _FakeQCRunner:
    """Minimal QCRunner fake.

    Purpose:
        Ensure QC is only run after a successful Copilot invocation.
    """

    call_log: list[str]

    def run_scoped(self) -> None:
        """Record that scoped QC ran."""

        self.call_log.append("qc")


def test_executor_does_not_flip_checkbox_until_throttle_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throttled Copilot call must not advance the plan until success."""

    from scripts.dev_tools.atomic_executor import cli

    call_log: list[str] = []

    # Patch logging helper to avoid writing log files during tests.
    monkeypatch.setattr(cli, "_log_msg", _noop_log)

    # Arrange a fake Copilot outcome sequence: throttle -> success.
    outcomes = [
        CopilotRunResult(exit_code=1, output_tail="HTTP 429 rate limit"),
        CopilotRunResult(exit_code=0, output_tail=""),
    ]

    def _fake_run_copilot(**_kwargs: object) -> CopilotRunResult:
        """Pop a pre-canned Copilot result without invoking any subprocess."""

        call_log.append("copilot")
        return outcomes.pop(0)

    monkeypatch.setattr(cli, "run_copilot", _fake_run_copilot)

    # Build a minimal task model.
    cur = PlanTask(
        task_id="P3-T2",
        phase=3,
        task_num=2,
        title="Add retry loop",
        checked=False,
        line_index=0,
    )

    # After Copilot runs, executor checks plan state and flips checkbox if needed.
    task_after = PlanTask(
        task_id=cur.task_id,
        phase=cur.phase,
        task_num=cur.task_num,
        title=cur.title,
        checked=False,
        line_index=0,
    )

    parser = _FakePlanParser(task_after=task_after, call_log=call_log)
    builder = _FakePromptBuilder(prompt_text="prompt")
    qc_runner = _FakeQCRunner(call_log=call_log)

    clock = _FakeClock(start=100.0)
    sleeper = _FakeSleeper(call_log)
    limiter = CallRateLimiter(
        max_calls=10, window_seconds=60.0, clock=clock, sleeper=sleeper
    )

    backoff = ExponentialBackoff(
        base_seconds=1.0,
        max_seconds=1.0,
        random_source=_FakeRandom(1.0),
    )

    # Act.
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
        copilot_max_retries=3,
        copilot_output_tail_bytes=256,
        copilot_allow_shell=True,
        copilot_allow_all_paths=True,
        copilot_allow_all_urls=False,
        copilot_trust_workspace=True,
        print_prompt=False,
        copy_prompt=False,
    )

    # Assert.
    assert exit_code == 0

    # A single backoff sleep must occur between the two Copilot invocations.
    assert sleeper.sleeps == [1.0]

    # Ordering invariants:
    # - Copilot is invoked twice (throttle then success).
    # - QC and checkbox flip happen only after Copilot succeeds.
    assert call_log.count("copilot") == 2
    assert "qc" in call_log
    assert any(item.startswith("flip:") for item in call_log)

    first_qc_index = call_log.index("qc")
    first_flip_index = next(
        idx for idx, item in enumerate(call_log) if item.startswith("flip:")
    )

    # Ensure QC/flip do not occur before the second Copilot invocation.
    # (Second invocation index is 1 because we log "copilot" for each call.)
    second_copilot_index = [
        idx for idx, item in enumerate(call_log) if item == "copilot"
    ][1]
    assert second_copilot_index < first_qc_index
    assert first_qc_index < first_flip_index

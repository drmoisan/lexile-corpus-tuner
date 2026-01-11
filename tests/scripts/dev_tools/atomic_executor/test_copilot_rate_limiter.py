"""Unit tests for call-rate limiter behavior (no real sleeping).

These tests define deterministic scheduling behavior for a "calls per window"
rate limiter used by the atomic executor when invoking Copilot CLI.

Note:
    This module is expected to FAIL until the rate limiter implementation is
    introduced (issue #80).
"""

from __future__ import annotations


class _FakeClock:
    """Deterministic monotonic clock for tests."""

    def __init__(self, start: float) -> None:
        self._now = start

    def now_monotonic(self) -> float:
        """Return the current monotonic time."""

        return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by a positive duration."""

        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self._now += seconds


class _FakeSleeper:
    """Sleeper that records delays and advances the fake clock."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        """Record the sleep duration and advance the clock."""

        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self.sleeps.append(seconds)
        self._clock.advance(seconds)


def test_rate_limiter_allows_calls_under_limit_without_sleep() -> None:
    """Under the max-calls threshold, the limiter should not sleep."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import CallRateLimiter

    clock = _FakeClock(start=100.0)
    sleeper = _FakeSleeper(clock)

    limiter = CallRateLimiter(
        max_calls=2, window_seconds=10.0, clock=clock, sleeper=sleeper
    )

    limiter.acquire()
    clock.advance(1.0)
    limiter.acquire()

    assert sleeper.sleeps == []


def test_rate_limiter_sleeps_until_oldest_call_expires() -> None:
    """When at the limit, a new acquire sleeps until the window permits it."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import CallRateLimiter

    clock = _FakeClock(start=100.0)
    sleeper = _FakeSleeper(clock)

    limiter = CallRateLimiter(
        max_calls=2, window_seconds=10.0, clock=clock, sleeper=sleeper
    )

    limiter.acquire()  # t=100
    clock.advance(1.0)
    limiter.acquire()  # t=101

    # At t=102, the oldest call (t=100) is still within the 10s window.
    clock.advance(1.0)

    limiter.acquire()

    assert sleeper.sleeps == [8.0]
    assert clock.now_monotonic() == 110.0

"""Unit tests for exponential backoff behavior (deterministic jitter).

These tests specify the backoff schedule used when Copilot CLI throttling is
classified as retryable.

Note:
    This module is expected to FAIL until the backoff implementation is
    introduced (issue #80).
"""

from __future__ import annotations


class _FakeRandom:
    """Deterministic random source returning a fixed value."""

    def __init__(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("value must be in [0, 1]")
        self._value = value

    def random(self) -> float:
        """Return the fixed random value."""

        return self._value


def test_backoff_grows_exponentially_with_full_jitter() -> None:
    """Delays should grow as base*2^k with full jitter, capped by max."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import ExponentialBackoff

    backoff = ExponentialBackoff(
        base_seconds=2.0,
        max_seconds=60.0,
        random_source=_FakeRandom(0.5),
    )

    # First few throttle events should yield: 2, 4, 8 nominal, each with 0.5 jitter.
    assert backoff.on_throttle() == 1.0
    assert backoff.on_throttle() == 2.0
    assert backoff.on_throttle() == 4.0


def test_backoff_caps_at_max_seconds() -> None:
    """Backoff must cap at max_seconds before jitter is applied."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import ExponentialBackoff

    backoff = ExponentialBackoff(
        base_seconds=50.0,
        max_seconds=60.0,
        random_source=_FakeRandom(0.5),
    )

    # First throttle: nominal 50 -> jitter 25.
    assert backoff.on_throttle() == 25.0

    # Second throttle: nominal would be 100, but cap to 60 -> jitter 30.
    assert backoff.on_throttle() == 30.0


def test_backoff_resets_after_success() -> None:
    """A non-throttle success should reset/decay the backoff state."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import ExponentialBackoff

    backoff = ExponentialBackoff(
        base_seconds=2.0,
        max_seconds=60.0,
        random_source=_FakeRandom(0.5),
    )

    assert backoff.on_throttle() == 1.0
    assert backoff.on_throttle() == 2.0

    backoff.on_success()

    # After reset, the next throttle should behave like the first.
    assert backoff.on_throttle() == 1.0

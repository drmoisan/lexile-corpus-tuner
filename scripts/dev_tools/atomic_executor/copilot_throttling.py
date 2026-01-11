"""Deterministic throttling primitives for Copilot CLI invocation.

Purpose:
    The atomic executor invokes the Copilot CLI once per atomic task attempt.
    When invoked too quickly, the CLI or upstream services can respond with
    throttling/rate-limit errors.

    This module provides small, fully typed primitives that allow the executor
    to self-regulate *call frequency* (calls per time window) and to apply
    throttle-aware retries with bounded exponential backoff.

Design constraints:
    - Must be deterministic and unit-testable.
    - Must not require real sleeps in tests (inject a sleeper).
    - Must not require real time in tests (inject a clock).

Note:
    Higher-level orchestration (when to retry, how many times, and how to surface
    errors) lives in the executor CLI layer.
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Clock(Protocol):
    """Clock abstraction for deterministic time in tests.

    Purpose:
        `time.monotonic()` is not deterministic in tests. The rate limiter and
        backoff policy depend on monotonic time, so we inject a clock.
    """

    def now_monotonic(self) -> float:
        """Return a monotonic timestamp in seconds."""
        ...


class Sleeper(Protocol):
    """Sleep abstraction for deterministic scheduling in tests.

    Purpose:
        Production code can call `time.sleep`, but tests must not do real sleeps.
        Injecting a sleeper allows tests to record and simulate time advances.
    """

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration (seconds)."""


class RandomSource(Protocol):
    """Random source abstraction for deterministic jitter in tests.

    Purpose:
        Production uses a system randomness source. Tests must be able to provide a
        deterministic value so backoff behavior can be asserted precisely.
    """

    def random(self) -> float:
        """Return a random float in the interval [0.0, 1.0]."""
        ...


def _new_call_times() -> deque[float]:
    """Create an empty typed deque for CallRateLimiter timestamps.

    Purpose:
        `collections.deque` is not generic at runtime, so `field(default_factory=deque)`
        leads Pyright to infer `deque[Unknown]`. Providing a typed factory keeps the
        dataclass field fully typed under strict settings.

    Returns:
        deque[float]: Empty deque to store monotonic call timestamps.
    """

    return deque()


@dataclass
class CallRateLimiter:
    """Enforce a maximum number of calls per sliding time window.

    Purpose:
        Prevent rapid-fire Copilot CLI invocations that are likely to trigger
        throttling. This limiter is strictly *call-count* based; it intentionally
        does not attempt to estimate token usage.

    Usage:
        Call `acquire()` immediately before invoking Copilot.

    Invariants / Constraints:
        - `max_calls` must be >= 1.
        - `window_seconds` must be > 0.
        - Uses monotonic time (via `clock`) so it is unaffected by wall-clock
          changes.

    Side Effects:
        May call `sleeper.sleep(...)` when the call budget is exhausted.

    Attributes:
        max_calls (int): Maximum number of calls permitted in the trailing window.
        window_seconds (float): Window size in seconds.
        clock (Clock): Injected clock source.
        sleeper (Sleeper): Injected sleeper used for scheduling.
    """

    max_calls: int
    window_seconds: float
    clock: Clock
    sleeper: Sleeper

    _call_times: deque[float] = field(default_factory=_new_call_times)

    def acquire(self) -> None:
        """Block (via injected sleeper) until a call is permitted, then record it.

        Purpose:
            Enforces a sliding-window rate limit. When a call would exceed the
            `max_calls` budget, the limiter sleeps until the oldest call falls
            outside the window.

        Raises:
            ValueError: If configured with invalid parameters.

        Side Effects:
            May sleep (via `sleeper`) and always records a new call timestamp.
        """

        if self.max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        while True:
            now = self.clock.now_monotonic()
            window_start = now - self.window_seconds

            # Evict timestamps that are no longer within the trailing window.
            while self._call_times and self._call_times[0] <= window_start:
                self._call_times.popleft()

            # Under the limit: record immediately.
            if len(self._call_times) < self.max_calls:
                self._call_times.append(now)
                return

            # At limit: sleep until the oldest call expires.
            oldest = self._call_times[0]
            sleep_seconds = (oldest + self.window_seconds) - now

            # Guard against clock/sleeper anomalies: treat non-positive delays as
            # "no delay" and re-evaluate.
            if sleep_seconds > 0:
                self.sleeper.sleep(sleep_seconds)


@dataclass
class ExponentialBackoff:
    """Exponential backoff with full jitter and a hard cap.

    Purpose:
        When a Copilot invocation is classified as throttled/rate-limited, the
        executor should pause before retrying. Exponential backoff reduces the
        chance of repeated throttling while preserving forward progress.

    Backoff model:
        Let $k$ be the number of throttle events observed since the last
        successful invocation. The nominal delay is:

        $$\\text{nominal} = \\text{base} \\cdot 2^k$$

        The nominal delay is capped to `max_seconds` *before* jitter.
        Full jitter means sampling uniformly from [0, cap]. With an injected
        `RandomSource`, we compute:

        $$\\text{delay} = U(0,1) \\cdot \\min(\\text{max}, \\text{nominal})$$

    Invariants / Constraints:
        - `base_seconds` must be > 0.
        - `max_seconds` must be > 0.

    Side Effects:
        None. This class only computes delays; it does not sleep.

    Attributes:
        base_seconds (float): Base delay in seconds.
        max_seconds (float): Maximum delay cap in seconds.
        random_source (RandomSource): Injected random provider.
    """

    base_seconds: float
    max_seconds: float
    random_source: RandomSource

    _throttle_count: int = 0

    def on_throttle(self) -> float:
        """Record a throttle event and return the next delay in seconds.

        Returns:
            float: Backoff delay in seconds (may be 0.0 with full jitter).

        Raises:
            ValueError: If configured with invalid parameters.
        """

        if self.base_seconds <= 0:
            raise ValueError("base_seconds must be > 0")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")

        nominal = self.base_seconds * (2**self._throttle_count)
        self._throttle_count += 1

        capped = self.max_seconds if nominal > self.max_seconds else nominal
        jitter = self.random_source.random()

        # The RandomSource contract is [0, 1], but we clamp defensively to keep
        # backoff bounded even if a caller provides a buggy implementation.
        if jitter < 0.0:
            jitter = 0.0
        elif jitter > 1.0:
            jitter = 1.0

        return jitter * capped

    def on_success(self) -> None:
        """Reset backoff state after a successful (non-throttle) invocation."""

        self._throttle_count = 0


class FailureKind(Enum):
    """Classification of a Copilot invocation failure.

    Purpose:
        The executor needs to decide whether a failure is retryable (throttle)
        or should fail fast (non-throttle).
    """

    THROTTLE = "throttle"
    NON_THROTTLE = "non_throttle"


def classify_copilot_failure(*, exit_code: int, output_tail: str) -> FailureKind:
    """Classify a Copilot CLI outcome as throttle vs non-throttle.

    Purpose:
        Copilot CLI failures are surfaced primarily via an exit code and the
        streamed output written to the executor log. This classifier uses a
        bounded output tail snippet so it does not depend on
        `CalledProcessError.stdout`/`stderr`.

    Args:
        exit_code (int): Copilot CLI process exit code.
        output_tail (str): Bounded output tail snippet (case-insensitive).

    Returns:
        FailureKind: THROTTLE when output suggests a retryable rate limit;
            otherwise NON_THROTTLE.
    """

    # Exit code 0 is always success (non-throttle).
    if exit_code == 0:
        return FailureKind.NON_THROTTLE

    tail = output_tail.lower()

    # Match common provider/service signals. Keep this simple and explicit so it
    # is easy to reason about in tests and logs.
    throttle_markers = (
        "rate limit",
        "rate limited",
        "throttle",
        "too many requests",
        "http 429",
        " 429",
        "429 ",
        "(429)",
        "http 503",
        " 503",
        "503 ",
        "(503)",
        "service unavailable",
    )

    # The decision is purely signature-based: if any marker appears in the tail,
    # we treat it as retryable throttling.
    for marker in throttle_markers:
        if marker in tail:
            return FailureKind.THROTTLE

    return FailureKind.NON_THROTTLE


@dataclass(frozen=True)
class SystemClock:
    """Production clock implementation using `time.monotonic()`."""

    def now_monotonic(self) -> float:
        """Return the system monotonic time in seconds."""

        return time.monotonic()


@dataclass(frozen=True)
class TimeSleeper:
    """Production sleeper implementation using `time.sleep()`."""

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""

        time.sleep(seconds)


@dataclass(frozen=True)
class SystemRandom:
    """Production random source using `secrets.randbits()` (crypto-strong)."""

    def random(self) -> float:
        """Return a random float in the interval [0.0, 1.0]."""

        # Produce a uniform-ish float in [0.0, 1.0) using a cryptographically
        # strong source to satisfy security linting (even though this jitter is
        # not used for secrets).
        return secrets.randbits(53) / (1 << 53)

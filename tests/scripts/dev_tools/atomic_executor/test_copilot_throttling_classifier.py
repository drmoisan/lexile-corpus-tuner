"""Unit tests for Copilot CLI throttling classification.

These tests define the expected behavior for classifying Copilot CLI failures as
"throttle" vs "non-throttle" based on exit code and an output tail snippet.

They are intentionally deterministic and do not invoke subprocesses or the
network.

Note:
    This module is expected to FAIL until the production classifier is
    implemented (issue #80).
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("exit_code", "output_tail"),
    [
        (1, "Error: rate limit exceeded"),
        (1, "RATE LIMITED: please retry"),
        (1, "HTTP 429 Too Many Requests"),
        (1, "Service Unavailable (503)"),
        (1, "throttle detected, backing off"),
    ],
)
def test_classify_throttle_positive_samples(exit_code: int, output_tail: str) -> None:
    """Throttle-like tails should classify as THROTTLE."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import (
        FailureKind,
        classify_copilot_failure,
    )

    assert classify_copilot_failure(exit_code=exit_code, output_tail=output_tail) is (
        FailureKind.THROTTLE
    )


@pytest.mark.parametrize(
    ("exit_code", "output_tail"),
    [
        (1, "Permission denied: cannot execute"),
        (1, "Authentication failed: invalid token"),
        (1, "fatal: not a git repository"),
        (1, "Unknown error occurred"),
        (1, "Traceback (most recent call last): ..."),
    ],
)
def test_classify_throttle_negative_samples(exit_code: int, output_tail: str) -> None:
    """Non-throttle failures should classify as NON_THROTTLE."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import (
        FailureKind,
        classify_copilot_failure,
    )

    assert classify_copilot_failure(exit_code=exit_code, output_tail=output_tail) is (
        FailureKind.NON_THROTTLE
    )


def test_classify_success_exit_code_is_non_throttle() -> None:
    """Exit code 0 should never classify as THROTTLE."""

    from scripts.dev_tools.atomic_executor.copilot_throttling import (
        FailureKind,
        classify_copilot_failure,
    )

    assert classify_copilot_failure(exit_code=0, output_tail="rate limit") is (
        FailureKind.NON_THROTTLE
    )

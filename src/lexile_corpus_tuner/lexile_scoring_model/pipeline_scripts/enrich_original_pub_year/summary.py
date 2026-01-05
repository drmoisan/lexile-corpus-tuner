from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .match_result import MatchResult


class Summary:
    """Track high/low/none matches and errors for reporting and checkpoints."""

    def __init__(self) -> None:
        self.matched_high = 0
        self.matched_low = 0
        self.matched_none = 0
        self.errors = 0

    def record(self, result: MatchResult) -> None:
        if result.confidence == "high":
            self.matched_high += 1
        elif result.confidence == "low":
            self.matched_low += 1
        else:
            self.matched_none += 1

    def record_error(self) -> None:
        self.errors += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "matched_high": self.matched_high,
            "matched_low": self.matched_low,
            "matched_none": self.matched_none,
            "errors": self.errors,
        }

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .match_result import MatchResult


class Summary:
    """
    Collects enrichment metrics for reporting and checkpoint persistence.

    Purpose:
        Track counts of high/low/none matches and errors to gauge enrichment quality.
    """

    def __init__(self) -> None:
        """
        Initialize counters for match confidence buckets and errors.

        Purpose:
            Start a fresh metrics accumulator for a single enrichment run.

        Raises:
            None

        Side Effects:
            None.
        """

        self.matched_high = 0
        self.matched_low = 0
        self.matched_none = 0
        self.errors = 0

    def record(self, result: MatchResult) -> None:
        """
        Increment counters based on the confidence level of a match result.

        Args:
            result (MatchResult): Match decision to classify.

        Purpose:
            Maintain counts of match quality for reporting and checkpoints.

        Raises:
            None

        Side Effects:
            Mutates in-memory counters.
        """

        # Bucket counts by confidence so downstream reporting surfaces quality.
        if result.confidence == "high":
            self.matched_high += 1
        elif result.confidence == "low":
            self.matched_low += 1
        else:
            self.matched_none += 1

    def record_error(self) -> None:
        """
        Increment the error counter when a lookup fails permanently.

        Purpose:
            Track unrecoverable errors so reporting reflects failure rates.

        Raises:
            None

        Side Effects:
            Mutates in-memory counters.
        """

        self.errors += 1

    def to_dict(self) -> dict[str, int]:
        """
        Return metrics as a dict for serialization or logging.

        Purpose:
            Provide a serialization-friendly view of the accumulated metrics.

        Returns:
            dict[str, int]: Current metrics keyed by label.

        Raises:
            None

        Side Effects:
            None.
        """

        return {
            "matched_high": self.matched_high,
            "matched_low": self.matched_low,
            "matched_none": self.matched_none,
            "errors": self.errors,
        }

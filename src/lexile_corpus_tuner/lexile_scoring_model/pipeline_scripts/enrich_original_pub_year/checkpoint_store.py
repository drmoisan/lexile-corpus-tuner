from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .summary import Summary


class CheckpointStore(Protocol):
    """
    Interface for persisting progress so enrichment can resume safely.

    Purpose:
        Provide a common API for saving and loading the last processed index and
        summary metrics.

    Usage:
        Implementations may use files, databases, or other stores, but callers
        interact only through this protocol.

    Invariants / Constraints:
        Stored checkpoints must map to row indexes in the processed dataframe and
        remain durable across process restarts.

    Side Effects:
        Implementations may perform disk or network I/O during load and save.
    """

    def load(self) -> int:
        """
        Load the last processed row index from durable storage.

        Purpose:
            Resume enrichment without reprocessing rows already handled.

        Returns:
            int: Zero when no checkpoint exists, otherwise the next row index to
            resume from.

        Raises:
            Exception: Implementations may surface I/O or deserialization errors.

        Side Effects:
            May perform disk or network I/O when reading persisted checkpoints.
        """

        ...

    def save(self, index: int, summary: Summary) -> None:
        """
        Persist the latest processed index along with summary metrics.

        Purpose:
            Enable resumable enrichment and reporting continuity across runs.

        Args:
            index (int): Row index to resume from in the next run.
            summary (Summary): Aggregated enrichment metrics to persist.

        Raises:
            Exception: Implementations may surface I/O or serialization errors.

        Side Effects:
            May perform disk or network I/O when writing checkpoints.
        """

        ...

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .match_result import MatchResult


class CacheStore(Protocol):
    """
    Abstraction for persisting match results so enrichment can reuse previous work.

    Purpose:
        Provide a stable interface for storing and retrieving `MatchResult` values
        keyed by normalized title/author pairs.

    Usage:
        Implementations may use files, databases, or in-memory stores without
        requiring callers to change enrichment flow.

    Invariants / Constraints:
        Keys must be deterministic for a given title/author combination so cache
        hits remain stable between runs.

    Side Effects:
        Implementations may perform I/O when persisting or reading cached entries.
    """

    def get(self, key: str) -> MatchResult | None:
        """
        Retrieve a previously stored match result for the given cache key.

        Purpose:
            Provide a consistent lookup contract for cache implementations.

        Args:
            key (str): Normalized identifier built from title and author.

        Returns:
            MatchResult | None: Cached match if present; otherwise `None`.

        Raises:
            Exception: Implementations may surface I/O or deserialization errors.

        Side Effects:
            Implementations may perform disk or network I/O when fetching values.
        """

        ...

    def set(self, key: str, value: MatchResult) -> None:
        """
        Persist a match result under a deterministic cache key.

        Purpose:
            Allow enrichment to store decisions for reuse without coupling to storage.

        Args:
            key (str): Normalized identifier built from title and author.
            value (MatchResult): Match decision to persist for reuse.

        Raises:
            Exception: Implementations may surface I/O or serialization errors.

        Side Effects:
            Implementations may perform disk or network I/O when writing values.
        """

        ...

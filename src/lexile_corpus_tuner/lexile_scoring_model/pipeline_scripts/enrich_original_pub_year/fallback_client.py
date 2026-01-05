from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .match_candidate import MatchCandidate


class FallbackClient(Protocol):
    """
    Optional secondary catalog client used when Open Library returns low confidence.

    Purpose:
        Allow enrichment to query alternate sources such as Wikidata or the Library
        of Congress without hard-coding a specific provider.

    Usage:
        Implementations supply a `search` method that returns match candidates for a
        title and author when primary lookup yields no confident match.

    Invariants / Constraints:
        Implementations should return deterministic results for the same inputs to
        keep caching effective.

    Side Effects:
        Implementations may perform network I/O and should respect rate limits.
    """

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        """
        Query an alternate catalog for candidate publication years.

        Purpose:
            Supply additional candidates when the primary provider cannot find a
            confident match.

        Args:
            title (str): Raw title from the dataset row.
            author (str): Raw author string from the dataset row.

        Returns:
            list[MatchCandidate]: Possible matches ordered by provider relevance.

        Raises:
            Exception: Implementations may surface network or parsing errors.

        Side Effects:
            Implementations may perform network calls and apply their own caching.
        """

        ...

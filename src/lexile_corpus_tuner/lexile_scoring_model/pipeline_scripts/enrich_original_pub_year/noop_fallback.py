from __future__ import annotations

from typing import TYPE_CHECKING

from .fallback_client import FallbackClient

if TYPE_CHECKING:
    from .match_candidate import MatchCandidate


class NoopFallback(FallbackClient):
    """
    Fallback implementation that intentionally yields no candidates.

    Purpose:
        Provide a safe default when optional catalog lookups are disabled.
    """

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        """
        Return an empty candidate list to indicate no fallback matches are available.

        Purpose:
            Serve as a safe default when no alternate catalog providers are enabled.

        Args:
            title (str): Unused title input.
            author (str): Unused author input.

        Returns:
            list[MatchCandidate]: Always empty.

        Raises:
            None

        Side Effects:
            None.
        """

        return []

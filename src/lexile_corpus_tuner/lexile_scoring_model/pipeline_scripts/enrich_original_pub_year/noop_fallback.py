from __future__ import annotations

from typing import TYPE_CHECKING

from .fallback_client import FallbackClient

if TYPE_CHECKING:
    from .match_candidate import MatchCandidate


class NoopFallback(FallbackClient):
    """Return no results when optional sources are disabled."""

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        return []

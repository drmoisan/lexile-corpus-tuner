from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .match_candidate import MatchCandidate


class FallbackClient(Protocol):
    """Protocol for optional secondary catalog clients (e.g., Wikidata or LOC)."""

    def search(self, title: str, author: str) -> list[MatchCandidate]: ...

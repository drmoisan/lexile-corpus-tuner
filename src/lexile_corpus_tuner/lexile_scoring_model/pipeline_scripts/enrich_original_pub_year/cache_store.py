from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .match_result import MatchResult


class CacheStore(Protocol):
    """Cache protocol so storage backends can be swapped without changing logic."""

    def get(self, key: str) -> MatchResult | None: ...

    def set(self, key: str, value: MatchResult) -> None: ...

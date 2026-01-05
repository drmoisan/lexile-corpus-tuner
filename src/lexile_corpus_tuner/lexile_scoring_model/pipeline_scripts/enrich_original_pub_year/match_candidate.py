from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchCandidate:
    """Raw search hit from a provider before confidence selection."""

    title: str
    author: str
    year: int | None
    source: str
    score: float

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchCandidate:
    """
    Single search hit returned by a provider prior to scoring and selection.

    Purpose:
        Represent raw provider responses with enough metadata for downstream
        filtering and confidence assessment.

    Attributes:
        title (str): Title string as returned by the provider.
        author (str): Author string as returned by the provider.
        year (int | None): Candidate publication year if supplied.
        source (str): Provider identifier (e.g., `openlibrary`).
        score (float): Provider-native ranking score; may be zero when unknown.
    """

    title: str
    author: str
    year: int | None
    source: str
    score: float

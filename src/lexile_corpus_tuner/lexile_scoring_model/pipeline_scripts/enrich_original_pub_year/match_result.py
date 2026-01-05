from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Confidence = Literal["high", "low", "none"]


@dataclass
class MatchResult:
    """
    Final decision for a publication year lookup, with confidence and provenance.

    Purpose:
        Convey the chosen year (or absence) along with how confident the pipeline is
        in that choice and where it came from.

    Attributes:
        year (int | None): Selected publication year; `None` when unmatched.
        confidence (Confidence): `"high"`, `"low"`, or `"none"` depending on match
        quality.
        source (str | None): Identifier for the provider or fallback that supplied
        the match.
    """

    year: int | None
    confidence: Confidence
    source: str | None

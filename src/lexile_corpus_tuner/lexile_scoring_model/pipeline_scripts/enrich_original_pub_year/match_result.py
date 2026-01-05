from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Confidence = Literal["high", "low", "none"]


@dataclass
class MatchResult:
    """Final decision about a matched publication year and its confidence level."""

    year: int | None
    confidence: Confidence
    source: str | None

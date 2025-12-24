from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "source_id",
    "text_id",
    "tokens",
    "genre",
    "era_bucket",
    "intended_audience",
)

OPTIONAL_FIELDS: tuple[str, ...] = (
    "publication_year",
    "grade_band",
    "weight",
)


@dataclass(slots=True)
class NormalizedDocument:
    source_id: str
    text_id: str
    tokens: list[str]
    genre: str
    era_bucket: str
    intended_audience: str
    publication_year: int | None = None
    grade_band: str | None = None
    weight: float | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> NormalizedDocument:
        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(
            source_id=str(raw["source_id"]),
            text_id=str(raw["text_id"]),
            tokens=list(raw.get("tokens", [])),
            genre=str(raw.get("genre", "")),
            era_bucket=str(raw.get("era_bucket", "")),
            intended_audience=str(raw.get("intended_audience", "")),
            publication_year=_coerce_int_or_none(raw.get("publication_year")),
            grade_band=_coerce_str_or_none(raw.get("grade_band")),
            weight=_coerce_float_or_none(raw.get("weight")),
        )


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)

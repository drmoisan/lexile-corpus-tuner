from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    """Represents an input document."""

    doc_id: str
    text: str


@dataclass(slots=True)
class Token:
    """Represents a token and its inclusive-exclusive character offsets."""

    text: str
    start_char: int
    end_char: int


@dataclass(slots=True)
class Window:
    """A contiguous run of tokens from a document."""

    doc_id: str
    window_id: int
    start_token_idx: int
    end_token_idx: int
    text: str


@dataclass(slots=True)
class WindowScore:
    """Lexile-like score for a window."""

    window: Window
    lexile: float


@dataclass(slots=True)
class DocumentLexileStats:
    """Computed Lexile statistics for a full document."""

    doc_id: str
    avg_lexile: float
    max_lexile: float
    window_scores: list[WindowScore]


@dataclass(slots=True)
class ConstraintViolation:
    """Represents a broken Lexile constraint."""

    doc_id: str
    window_id: int
    lexile: float
    reason: str
    start_token_idx: int
    end_token_idx: int

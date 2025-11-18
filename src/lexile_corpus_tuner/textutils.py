from __future__ import annotations

import re
import unicodedata
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*", re.UNICODE)


def normalize_text(value: object) -> str:
    """Normalize arbitrary text so corpus + analyzer share identical tokens."""
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def iter_tokens(value: str) -> Iterable[str]:
    """Yield tokens from normalized text using the shared TOKEN_RE."""
    normalized = normalize_text(value)
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)

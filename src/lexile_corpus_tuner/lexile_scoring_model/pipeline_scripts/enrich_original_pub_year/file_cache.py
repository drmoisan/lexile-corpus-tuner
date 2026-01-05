from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .cache_store import CacheStore
from .match_result import MatchResult

if TYPE_CHECKING:
    from pathlib import Path


class FileCache(CacheStore):
    """Simple JSON-on-disk cache keyed by normalized title and author."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        return self._cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> MatchResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return MatchResult(
                year=data.get("year"),
                confidence=data.get("confidence", "none"),
                source=data.get("source"),
            )
        except Exception:
            return None

    def set(self, key: str, value: MatchResult) -> None:
        path = self._path(key)
        payload = {
            "year": value.year,
            "confidence": value.confidence,
            "source": value.source,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

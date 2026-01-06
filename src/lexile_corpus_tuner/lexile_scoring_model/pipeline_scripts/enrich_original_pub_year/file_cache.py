from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .cache_store import CacheStore
from .match_result import MatchResult

if TYPE_CHECKING:
    from pathlib import Path


class FileCache(CacheStore):
    """
    Filesystem-backed cache storing match results as JSON for deterministic reuse.

    Purpose:
        Persist enrichment decisions between runs so repeated lookups avoid network
        calls.

    Usage:
        Initialized with a cache directory; provides `get`/`set` matching the
        `CacheStore` protocol.

    Side Effects:
        Performs filesystem I/O for every read/write.
    """

    def __init__(self, cache_dir: Path) -> None:
        """
        Create a cache rooted at the provided directory, creating it when absent.

        Purpose:
            Ensure the cache directory exists before any read/write operations.

        Args:
            cache_dir (Path): Directory where JSON cache files will be stored.

        Raises:
            OSError: If the cache directory cannot be created.

        Side Effects:
            Creates directories on disk when they do not already exist.
        """

        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """
        Build a safe, filesystem-friendly path for a cache key.

        Purpose:
            Normalize cache keys into file system paths that avoid invalid characters.

        Args:
            key (str): Normalized cache key derived from title and author.

        Returns:
            Path: Target path for the cached entry.

        Raises:
            None

        Side Effects:
            None.
        """

        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        return self._cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> MatchResult | None:
        """
        Load a cached match result if present; ignore unreadable entries.

        Purpose:
            Allow enrichment to reuse previously computed matches and skip requests.

        Args:
            key (str): Normalized cache key derived from title and author.

        Returns:
            MatchResult | None: Cached result, or `None` when missing or invalid.

        Raises:
            OSError: When the cache file cannot be opened.
            json.JSONDecodeError: When the file contents cannot be parsed.

        Side Effects:
            Performs disk reads; tolerates malformed JSON by treating it as a miss.
        """

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
        """
        Persist a match result to disk for future lookups.

        Purpose:
            Store computed matches so future runs can avoid redundant work.

        Args:
            key (str): Normalized cache key derived from title and author.
            value (MatchResult): Match decision to store.

        Raises:
            OSError: When the cache file cannot be written.
            TypeError: When payload cannot be serialized to JSON.

        Side Effects:
            Writes JSON to disk, overwriting any existing cached entry.
        """

        path = self._path(key)
        payload = {
            "year": value.year,
            "confidence": value.confidence,
            "source": value.source,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import requests

from .constants import OPEN_LIBRARY_URL
from .match_candidate import MatchCandidate

if TYPE_CHECKING:
    from .http_client import HttpClient


class OpenLibraryClient:
    """HTTP client for Open Library search with polite rate limiting and retries."""

    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        rate_limit: float = 5.0,
        timeout_seconds: float = 10.0,
        max_retries: int = 5,
        backoff_initial: float = 0.5,
        backoff_cap: float = 8.0,
    ) -> None:
        self._http = http or requests.Session()
        self._rate_limit = rate_limit
        self._timeout_seconds = timeout_seconds
        self._last_request = 0.0
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_cap = backoff_cap

    def _respect_rate_limit(self) -> None:
        if self._rate_limit <= 0:
            return
        min_interval = 1.0 / self._rate_limit
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        params = {"title": title, "author": author, "limit": "5"}
        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self._http.get(
                    OPEN_LIBRARY_URL, params=params, timeout=self._timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = min(
                    self._backoff_initial * (2 ** (attempt - 1)), self._backoff_cap
                )
                time.sleep(delay)
        if not isinstance(payload, dict):
            return []

        payload_dict = cast(dict[str, object], payload)
        docs_raw_value = payload_dict.get("docs", [])
        docs_raw_value_list: list[object] = []
        if isinstance(docs_raw_value, list):
            docs_raw_value_list = cast(list[object], docs_raw_value)
        docs_raw: list[dict[str, object]] = []
        for doc_value in docs_raw_value_list:
            if not isinstance(doc_value, dict):
                continue
            doc: dict[str, object] = cast(dict[str, object], doc_value)
            docs_raw.append(doc)

        candidates: list[MatchCandidate] = []
        for doc in docs_raw:
            cand_title = str(doc.get("title", ""))
            author_list_raw = doc.get("author_name", [])
            author_items: list[object] = []
            if isinstance(author_list_raw, list):
                author_items = cast(list[object], author_list_raw)
            else:
                author_items.append(author_list_raw)
            author_list = [str(item) for item in author_items]
            authors = ", ".join(author_list)
            year_val = doc.get("first_publish_year")
            year = int(year_val) if isinstance(year_val, int) else None
            score = 0.0
            candidates.append(
                MatchCandidate(
                    title=cand_title,
                    author=authors,
                    year=year,
                    source="openlibrary",
                    score=score,
                )
            )
        return candidates

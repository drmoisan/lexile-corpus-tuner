from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import requests

from .constants import OPEN_LIBRARY_URL
from .match_candidate import MatchCandidate


class OpenLibrarySearchError(Exception):
    """
    Raised when Open Library search fails after all retry attempts.

    Purpose:
        Provide clear context about which title/author lookup failed and why.

    Attributes:
        title (str): The title being searched.
        author (str): The author being searched.
        attempts (int): Number of attempts made before failure.
        last_error (Exception): The underlying exception that caused the failure.
    """

    def __init__(
        self, title: str, author: str, attempts: int, last_error: Exception
    ) -> None:
        """
        Initialize the exception with search context.

        Args:
            title (str): Title being searched.
            author (str): Author being searched.
            attempts (int): Number of retry attempts made.
            last_error (Exception): The underlying error that triggered this.

        Side Effects:
            None.
        """
        self.title = title
        self.author = author
        self.attempts = attempts
        self.last_error = last_error
        message = (
            f"Open Library search failed after {attempts} attempts "
            f"for title='{title}', author='{author}': {last_error}"
        )
        super().__init__(message)


if TYPE_CHECKING:
    from .http_client import HttpClient


class OpenLibraryClient:
    """
    HTTP client for Open Library search with rate limiting and bounded retries.

    Purpose:
        Encapsulate network access and response parsing so enrichment can request
        candidate publication years without duplicating HTTP logic.

    Usage:
        Instantiate once per run (optionally injecting a custom HTTP client) and
        call `search` for each title/author pair.

    Flow:
        Applies rate limiting, retries with exponential backoff on errors, then
        converts the JSON payload into `MatchCandidate` objects.

    Invariants / Constraints:
        Rate limit values must be positive to avoid divide-by-zero, and retries are
        capped by `max_retries` to prevent infinite loops.

    Side Effects:
        Performs network I/O and sleeps to honor rate limits and backoff delays.
    """

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
        """
        Configure the client with rate limiting, timeouts, and retry settings.

        Purpose:
            Allow callers to tune network behavior and inject custom HTTP clients.

        Args:
            http (HttpClient | None): Custom HTTP client; defaults to
                `requests.Session`.
            rate_limit (float): Requests per second; values <= 0 disable limiting.
            timeout_seconds (float): Timeout per HTTP request.
            max_retries (int): Maximum attempts before surfacing the error.
            backoff_initial (float): Initial backoff delay in seconds.
            backoff_cap (float): Maximum backoff delay between retries.

        Raises:
            None

        Side Effects:
            None.
        """

        self._http = http or requests.Session()
        self._rate_limit = rate_limit
        self._timeout_seconds = timeout_seconds
        self._last_request = 0.0
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_cap = backoff_cap

    def _respect_rate_limit(self) -> None:
        """
        Sleep as needed to maintain the configured requests-per-second ceiling.

        Purpose:
            Enforce polite network behavior toward upstream providers.

        Raises:
            None

        Side Effects:
            Updates `_last_request` and may block the caller with `time.sleep`.
        """

        if self._rate_limit <= 0:
            return
        min_interval = 1.0 / self._rate_limit
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        """
        Look up candidate publication years for a given title and author.

        Purpose:
            Provide raw candidates from Open Library for downstream selection.

        Args:
            title (str): Title text from the dataset row.
            author (str): Author text from the dataset row.

        Returns:
            list[MatchCandidate]: Parsed results from Open Library.

        Raises:
            Exception: Propagates if all retry attempts fail.

        Side Effects:
            Performs network I/O and may sleep due to rate limiting or backoff.
        """

        params = {"title": title, "author": author, "limit": "5"}
        attempt = 0
        # Retry until a successful response or until the attempt budget is exhausted.
        while True:
            self._respect_rate_limit()
            try:
                response = self._http.get(
                    OPEN_LIBRARY_URL, params=params, timeout=self._timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:
                attempt += 1
                if attempt >= self._max_retries:
                    raise OpenLibrarySearchError(
                        title=title, author=author, attempts=attempt, last_error=exc
                    ) from exc
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
        # Normalize the docs payload into dictionaries we can safely traverse.
        docs_raw: list[dict[str, object]] = []
        for doc_value in docs_raw_value_list:
            if not isinstance(doc_value, dict):
                continue
            doc: dict[str, object] = cast(dict[str, object], doc_value)
            docs_raw.append(doc)

        candidates: list[MatchCandidate] = []
        # Convert raw documents into typed match candidates for downstream selection.
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

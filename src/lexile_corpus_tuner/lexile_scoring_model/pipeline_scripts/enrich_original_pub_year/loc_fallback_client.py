"""
Library of Congress fallback client for publication year lookup.

Purpose:
    Provide an optional secondary source when Open Library cannot supply a year.

Notes:
    This implementation issues a lightweight search against the Library of Congress
    Search API and attempts to extract the earliest year from returned results. It
    is intentionally conservative and will only emit candidates when a year can be
    parsed. Callers should treat the confidence as low.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .match_candidate import MatchCandidate

if TYPE_CHECKING:
    from requests import Response

    from .http_client import HttpClient


class LocFallbackClient:
    """Fallback client that queries the Library of Congress search API."""

    _URL = "https://www.loc.gov/search/"

    def __init__(
        self,
        *,
        http: HttpClient,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._http = http
        self._timeout_seconds = timeout_seconds

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        """Query LOC search and emit match candidates when a year can be parsed."""

        query = f"{title} {author}".strip()
        if not query:
            return []

        response: Response = self._http.get(
            self._URL,
            params={"fo": "json", "q": query, "c": "5"},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results_obj_raw = payload.get("results")
        candidates: list[MatchCandidate] = []
        if not isinstance(results_obj_raw, list):
            return candidates

        results_obj = cast(list[object], results_obj_raw)

        for item_value in results_obj:
            if not isinstance(item_value, dict):
                continue
            item = cast(dict[str, object], item_value)
            date_val = item.get("date")
            year: int | None = None
            if isinstance(date_val, str) and date_val.isdigit():
                year = int(date_val)
            elif isinstance(date_val, int):
                year = date_val
            if year is None:
                continue

            cand_title = str(item.get("title", title))
            creator_val = item.get("creator", author)
            cand_author = (
                creator_val if isinstance(creator_val, str) else str(creator_val)
            )
            candidates.append(
                MatchCandidate(
                    title=cand_title,
                    author=cand_author,
                    year=year,
                    source="loc",
                    score=0.0,
                )
            )

        return candidates

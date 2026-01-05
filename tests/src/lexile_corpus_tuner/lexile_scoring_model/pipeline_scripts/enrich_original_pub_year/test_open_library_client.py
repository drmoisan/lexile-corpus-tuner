from __future__ import annotations

from typing import Any, cast

import pytest  # noqa: TCH002 - pytest is required at runtime for fixtures
import requests
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)
from requests import Response

HttpClient = enrich.HttpClient
OpenLibraryClient = enrich.OpenLibraryClient


class _StubResponse(Response):
    def __init__(self, payload: Any):
        super().__init__()
        self._payload = payload

    def raise_for_status(self) -> None:  # type: ignore[override]
        return None

    def json(self, **_kwargs: object) -> Any:  # type: ignore[override]
        return self._payload


def test_search_honors_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(enrich.open_library_client.time, "sleep", sleep_calls.append)

    times = iter([0.0, 0.75])
    monkeypatch.setattr(
        enrich.open_library_client.time, "monotonic", lambda: next(times)
    )

    requests_seen: list[dict[str, str]] = []

    class StubHttp:
        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            requests_seen.append({"url": url, **params})
            return _StubResponse(
                {
                    "docs": [
                        {
                            "title": "Book",
                            "author_name": ["A"],
                            "first_publish_year": 1900,
                        }
                    ]
                }
            )

    client = OpenLibraryClient(
        http=cast(HttpClient, StubHttp()),
        rate_limit=2.0,
        timeout_seconds=1.0,
    )

    result = client.search("Book", "A")

    assert sleep_calls == [0.5]
    assert requests_seen == [
        {
            "url": enrich.OPEN_LIBRARY_URL,
            "title": "Book",
            "author": "A",
            "limit": "5",
        }
    ]
    assert result[0].year == 1900
    assert result[0].author == "A"


def test_search_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(enrich.open_library_client.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(enrich.open_library_client.time, "monotonic", lambda: 0.0)

    class StubHttp:
        def __init__(self) -> None:
            self.calls = 0

        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            self.calls += 1
            if self.calls == 1:
                raise requests.HTTPError("boom")
            return _StubResponse(
                {
                    "docs": [
                        {
                            "title": "Recovered",
                            "author_name": ["Author One", "Author Two"],
                            "first_publish_year": 2001,
                        }
                    ]
                }
            )

    http = StubHttp()
    client = OpenLibraryClient(
        http=cast(HttpClient, http),
        rate_limit=0.0,
        backoff_initial=0.25,
        max_retries=2,
    )

    result = client.search("Recovered", "Author One")

    assert http.calls == 2
    assert sleep_calls == [0.25]
    assert len(result) == 1
    assert result[0].year == 2001
    assert result[0].author == "Author One, Author Two"


def test_search_returns_empty_when_payload_not_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(enrich.open_library_client.time, "sleep", _sleep)
    monkeypatch.setattr(enrich.open_library_client.time, "monotonic", lambda: 0.0)

    class StubHttp:
        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            return _StubResponse(["unexpected"])

    client = OpenLibraryClient(
        http=cast(HttpClient, StubHttp()),
        rate_limit=0.0,
        timeout_seconds=1.0,
    )

    result = client.search("Title", "Author")

    assert result == []


def test_search_handles_non_list_authors_and_skips_bad_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(enrich.open_library_client.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(enrich.open_library_client.time, "monotonic", lambda: 0.0)

    class StubHttp:
        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            return _StubResponse(
                {
                    "docs": [
                        123,
                        {
                            "title": "Solo",
                            "author_name": "One Author",
                            "first_publish_year": "n/a",
                        },
                    ],
                }
            )

    client = OpenLibraryClient(
        http=cast(HttpClient, StubHttp()),
        rate_limit=0.0,
        timeout_seconds=1.0,
    )

    result = client.search("Solo", "Author")

    assert sleep_calls == []
    assert len(result) == 1
    assert result[0].author == "One Author"
    assert result[0].year is None


def test_search_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(enrich.open_library_client.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(enrich.open_library_client.time, "monotonic", lambda: 0.0)

    class FailingHttp:
        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            raise requests.HTTPError("boom")

    client = OpenLibraryClient(
        http=cast(HttpClient, FailingHttp()),
        rate_limit=0.0,
        timeout_seconds=1.0,
        max_retries=2,
        backoff_initial=0.25,
        backoff_cap=1.0,
    )

    with pytest.raises(requests.HTTPError):
        client.search("X", "Y")

    assert sleep_calls == [0.25]

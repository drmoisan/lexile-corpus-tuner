from __future__ import annotations

from typing import Any, cast

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import ck12_catalog


def test_fetch_catalog_page_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fetch_catalog_page should return JSON dict when the HTTP call succeeds.

    Purpose:
        Confirm the happy path performs an HTTP GET with the configured timeout
        and returns parsed JSON data without alteration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self._json_data: dict[str, Any] = {"books": []}

        def raise_for_status(self) -> None:
            """No-op to simulate a 200 OK status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return parsed JSON data."""
            return self._json_data

    captured: dict[str, object] = {}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_catalog.requests, "get", _fake_get)

    result = ck12_catalog.fetch_catalog_page("https://example.com/catalog")

    assert result == {"books": []}
    assert captured["url"] == "https://example.com/catalog"
    assert captured["headers"] == ck12_catalog.REQUEST_HEADERS
    assert captured["timeout"] == ck12_catalog.REQUEST_TIMEOUT_SECONDS


def test_fetch_catalog_page_sends_browser_headers_issue73(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    fetch_catalog_page should send browser-like headers to avoid 403 Forbidden.

    Purpose:
        Regression test for Issue #73 verifying that proper User-Agent and other
        browser headers are sent to prevent CK-12 from blocking requests with
        403 Forbidden responses. Now fetches JSON instead of HTML.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.

    Side Effects:
        Verifies headers include User-Agent, Accept, and other standard browser fields.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self._json_data: dict[str, Any] = {"books": []}

        def raise_for_status(self) -> None:
            """No-op to simulate a 200 OK status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return parsed JSON data."""
            return self._json_data

    captured: dict[str, object] = {}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResponse:
        """Capture headers and return success response."""
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr(ck12_catalog.requests, "get", _fake_get)

    # Call with the CK-12 Browse API endpoint
    catalog_url = ck12_catalog.DEFAULT_CK12_CATALOG_URL
    result = ck12_catalog.fetch_catalog_page(catalog_url)

    # Verify the fix: headers include User-Agent and other browser fields
    assert result == {"books": []}
    headers_obj = captured["headers"]
    assert isinstance(headers_obj, dict)
    headers = cast(dict[str, str], headers_obj)
    assert "User-Agent" in headers
    assert "Mozilla" in headers["User-Agent"]  # Browser-like User-Agent
    assert headers.get("Accept") == "application/json, text/plain, */*"
    assert headers.get("Referer") == "https://www.ck12.org/"
    assert headers.get("Origin") == "https://www.ck12.org"


def test_fetch_catalog_page_targets_browse_api_with_required_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    fetch_catalog_page should call the CK-12 Browse API with required JSON headers.

    Purpose:
        Ensure the fetch uses the Browse FlexBook endpoint mandated by the spec and
        includes the browser-like header set needed for anonymous JSON access.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self._json_data: dict[str, Any] = {"books": []}

        def raise_for_status(self) -> None:
            """No-op to simulate a 200 OK status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return parsed JSON data."""
            return self._json_data

    captured: dict[str, object] = {}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_catalog.requests, "get", _fake_get)

    browse_url = "https://www.ck12.org/flx/browse/flexbook?limit=200"
    result = ck12_catalog.fetch_catalog_page(browse_url)

    assert result == {"books": []}
    assert captured["url"] == browse_url

    headers_obj = captured["headers"]
    assert isinstance(headers_obj, dict)
    headers = cast(dict[str, str], headers_obj)
    assert headers.get("User-Agent", "").startswith("Mozilla/5.0")
    assert headers.get("Accept") == "application/json, text/plain, */*"
    assert headers.get("Referer") == "https://www.ck12.org/"
    assert headers.get("Origin") == "https://www.ck12.org"
    assert headers.get("Sec-Fetch-Dest") == "empty"
    assert headers.get("Sec-Fetch-Mode") == "cors"
    assert headers.get("Sec-Fetch-Site") == "same-origin"
    timeout_value = captured["timeout"]
    assert isinstance(timeout_value, int)
    assert timeout_value == ck12_catalog.REQUEST_TIMEOUT_SECONDS


def test_parse_catalog_json_returns_empty_for_empty_books_list() -> None:
    """
    parse_catalog_json returns empty list when JSON contains no books.

    Purpose:
        Verify that an empty or missing books array is handled gracefully,
        returning an empty list rather than raising an exception.

    Side Effects:
        None. Pure function test.
    """

    # Empty books array
    empty_catalog: dict[str, Any] = {"books": []}
    entries = ck12_catalog.parse_catalog_json(empty_catalog)
    assert entries == []
    assert len(entries) == 0


def test_parse_catalog_json_with_valid_books() -> None:
    """
    parse_catalog_json should return unique CatalogEntry objects for FlexBooks.

    Purpose:
        Validate JSON parsing, slug extraction, deduplication, and metadata
        extraction for well-formed CK-12 catalog JSON containing book objects.
    """

    catalog_json = {
        "books": [
            {
                "artifactID": 1001,
                "artifactType": "flexbook",
                "handle": "CK-12-Geometry-Concepts",
                "Title": "CK-12 Geometry Concepts",
                "Content_URL": "https://flexbooks.ck12.org/cbook/ck-12-geometry-concepts/",
                "Language_Code": "EN",
            },
            {
                "artifactID": 1002,
                "artifactType": "flexbook",
                "handle": "CK-12-Physics",
                "Title": "Physics FlexBook",
                "Content_URL": "https://www.ck12.org/book/CK-12-Physics/",
                "Language_Code": "EN",
            },
        ]
    }

    entries = ck12_catalog.parse_catalog_json(catalog_json)

    assert len(entries) == 2
    assert entries[0].identifier == "ck-12-geometry-concepts"
    assert entries[0].title == "CK-12 Geometry Concepts"
    assert entries[0].source_id == "ck12"
    assert entries[0].language == ["EN"]
    assert entries[1].identifier == "ck-12-physics"
    assert entries[1].title == "Physics FlexBook"
    assert entries[1].language == ["EN"]


def test_parse_catalog_json_missing_title_uses_slug() -> None:
    """
    parse_catalog_json should handle missing Title by falling back to the slug.

    Purpose:
        Ensure books lacking Title fields still produce CatalogEntry
        objects with stable identifiers and default optional metadata rather than
        raising errors during parsing.
    """

    catalog_json = {
        "books": [
            {
                "artifactID": 2001,
                "artifactType": "flexbook",
                "handle": "CK-12-Biology",
                "Content_URL": "https://flexbooks.ck12.org/cbook/ck-12-biology/",
                "Language_Code": "EN",
            }
        ]
    }

    entries = ck12_catalog.parse_catalog_json(catalog_json)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.identifier == "ck-12-biology"
    assert entry.title == "ck-12-biology"  # Falls back to slug
    assert entry.creator is None
    assert entry.year is None
    assert entry.language == ["EN"]
    assert entry.license_url is None


def test_parse_catalog_json_slug_generation_idempotent() -> None:
    """
    generate_stable_slug should preserve already slugified identifiers.

    Purpose:
        Ensure slug normalization used by parse_catalog_json does not mutate
        FlexBook slugs that already conform to the expected hyphenated format,
        keeping identifiers stable across repeated normalization passes.
    """

    expected_slug = "ck-12-earth-science"
    catalog_json = {
        "books": [
            {
                "artifactID": 3001,
                "artifactType": "flexbook",
                "handle": "CK-12-Earth-Science",
                "Title": "Earth Science",
                "Content_URL": f"https://flexbooks.ck12.org/cbook/{expected_slug}/",
                "Language_Code": "EN",
            }
        ]
    }

    entries = ck12_catalog.parse_catalog_json(catalog_json)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.identifier == expected_slug
    # Reapplying slug normalization should not alter an existing slug.
    assert ck12_catalog.generate_stable_slug(entry.identifier) == expected_slug


def test_parse_catalog_json_uses_handle_for_identifier() -> None:
    """
    parse_catalog_json should derive identifier from canonical handle when provided.

    Purpose:
        Ensure the Browse API shape with `handle` but no Content_URL still produces
        stable identifiers derived from the canonical handle string.
    """

    catalog_json = {
        "books": [
            {
                "artifactID": 4001,
                "artifactType": "flexbook",
                "handle": "CK-12-Physics-FlexBook-2.0",
                "title": "CK-12 Physics FlexBook 2.0",
                "Language_Code": "EN",
            }
        ]
    }

    entries = ck12_catalog.parse_catalog_json(catalog_json)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.identifier == "ck-12-physics-flexbook-2-0"
    assert entry.title == "CK-12 Physics FlexBook 2.0"
    assert entry.language == ["EN"]


def test_parse_catalog_json_filters_missing_required_fields() -> None:
    """
    parse_catalog_json should ignore items missing artifactID, artifactType, or handle.

    Purpose:
        Ensure the parser only emits entries when all required CK-12 Browse API
        fields are present, preventing partially-formed records from entering
        downstream pipelines.
    """

    catalog_json = {
        "books": [
            {
                "artifactID": 5001,
                "artifactType": "flexbook",
                "handle": "CK-12-Algebra-1",
                "title": "CK-12 Algebra 1",
                "Language_Code": "EN",
            },
            {
                "artifactType": "flexbook",
                "handle": "CK-12-Missing-ArtifactID",
                "title": "Should Skip - Missing artifactID",
            },
            {
                "artifactID": 5003,
                "handle": "CK-12-Missing-ArtifactType",
                "title": "Should Skip - Missing artifactType",
            },
            {
                "artifactID": 5004,
                "artifactType": "flexbook",
                "title": "Should Skip - Missing handle",
            },
        ]
    }

    entries = ck12_catalog.parse_catalog_json(catalog_json)

    assert len(entries) == 1
    only_entry = entries[0]
    assert only_entry.identifier == "ck-12-algebra-1"
    assert only_entry.title == "CK-12 Algebra 1"
    assert only_entry.language == ["EN"]


def test_generate_stable_slug_repeated_calls_stay_stable() -> None:
    """
    generate_stable_slug should produce the same output across repeated calls.

    Purpose:
        Verify slug normalization is idempotent even when an already-normalized
        slug is passed back through the helper, preventing identifier drift when
        multiple pipeline stages normalize the same value.
    """

    raw_identifier = "CK-12 Earth Science"

    first_slug = ck12_catalog.generate_stable_slug(raw_identifier)
    second_slug = ck12_catalog.generate_stable_slug(first_slug)

    assert first_slug == "ck-12-earth-science"
    assert second_slug == first_slug

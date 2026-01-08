from __future__ import annotations

from pathlib import Path

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import ck12_catalog


def _read_fixture_text(filename: str) -> str:
    """
    Load static HTML fixture content for catalog parsing tests.

    Purpose:
        Provide a reusable helper to keep test bodies focused on assertions while
        ensuring fixtures live alongside the tests for easy updates.

    Args:
        filename (str): Fixture filename located in this test module's directory.

    Returns:
        str: Contents of the fixture file.

    Raises:
        FileNotFoundError: If the requested fixture does not exist.
        OSError: If reading the fixture file fails.

    Side Effects:
        Reads fixture content from disk for use in tests.
    """

    fixture_path = Path(__file__).with_name(filename)
    return fixture_path.read_text(encoding="utf-8")


def test_fetch_catalog_page_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fetch_catalog_page should return HTML text when the HTTP call succeeds.

    Purpose:
        Confirm the happy path performs an HTTP GET with the configured timeout
        and returns the response text without alteration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self.text = "<html>catalog</html>"

        def raise_for_status(self) -> None:
            """No-op to simulate a 200 OK status."""
            return None

    captured: dict[str, object] = {}

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_catalog.requests, "get", _fake_get)

    result = ck12_catalog.fetch_catalog_page("https://example.com/catalog")

    assert result == "<html>catalog</html>"
    assert captured["url"] == "https://example.com/catalog"
    assert captured["timeout"] == ck12_catalog.REQUEST_TIMEOUT_SECONDS


def test_parse_catalog_rows_with_valid_html() -> None:
    """
    parse_catalog_rows should return unique CatalogEntry objects for FlexBook links.

    Purpose:
        Validate HTML parsing, slug extraction, deduplication, and title fallback
        for well-formed CK-12 catalog markup containing FlexBook anchors using a
        representative fixture.
    """

    html = _read_fixture_text("ck12_catalog_valid.html")

    entries = ck12_catalog.parse_catalog_rows(html)

    assert len(entries) == 2
    assert entries[0].identifier == "ck-12-geometry-concepts"
    assert entries[0].title == "CK-12 Geometry Concepts"
    assert entries[0].source_id == "ck12"
    assert entries[1].identifier == "ck-12-physics"
    assert entries[1].title == "Physics FlexBook"


def test_parse_catalog_rows_missing_title_uses_slug() -> None:
    """
    parse_catalog_rows should handle absent link text by falling back to the slug.

    Purpose:
        Ensure FlexBook anchors lacking display text still produce CatalogEntry
        objects with stable identifiers and default optional metadata rather than
        raising errors during parsing.
    """

    html = """
    <html>
        <body>
            <div class="content">
                <a href="/cbook/ck-12-biology"></a>
                <a href="/assets/logo.png">Ignore non-FlexBook asset</a>
            </div>
        </body>
    </html>
    """

    entries = ck12_catalog.parse_catalog_rows(html)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.identifier == "ck-12-biology"
    assert entry.title == "ck-12-biology"
    assert entry.creator is None
    assert entry.year is None
    assert entry.language == []
    assert entry.license_url is None


def test_parse_catalog_rows_slug_generation_idempotent() -> None:
    """
    generate_stable_slug should preserve already slugified identifiers.

    Purpose:
        Ensure slug normalization used by parse_catalog_rows does not mutate
        FlexBook slugs that already conform to the expected hyphenated format,
        keeping identifiers stable across repeated normalization passes.
    """

    expected_slug = "ck-12-earth-science"
    html = f"""
    <html>
        <body>
            <div class="content">
                <a href="/cbook/{expected_slug}/">Earth Science</a>
            </div>
        </body>
    </html>
    """

    entries = ck12_catalog.parse_catalog_rows(html)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.identifier == expected_slug
    # Reapplying slug normalization should not alter an existing slug.
    assert ck12_catalog.generate_stable_slug(entry.identifier) == expected_slug


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

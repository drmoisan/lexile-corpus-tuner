from __future__ import annotations

from pathlib import Path

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    ck12_enrichment,
)


def test_fetch_flexbook_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fetch_flexbook_html should return HTML text when the HTTP call succeeds.

    Purpose:
        Confirm the happy path performs an HTTP GET with the configured timeout,
        browser headers, and returns the response text without alteration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self.status_code = 200
            self.text = "<html>flexbook</html>"

    captured: dict[str, object] = {}

    def _fake_get(
        url: str, headers: dict[str, str] | None = None, timeout: int | None = None
    ) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_enrichment.requests, "get", _fake_get)

    result = ck12_enrichment.fetch_flexbook_html("https://example.com/flexbook")

    assert result == "<html>flexbook</html>"
    assert captured["url"] == "https://example.com/flexbook"
    assert captured["headers"] == ck12_enrichment.REQUEST_HEADERS
    assert captured["timeout"] == ck12_enrichment.REQUEST_TIMEOUT_SECONDS


def test_parse_flexbook_metadata_extracts_author_grade_language() -> None:
    """
    parse_flexbook_metadata should pull author, grade, and language from JSON-LD.

    Purpose:
        Validate that structured JSON-LD blocks populate the expected metadata
        fields, including normalization of the language code.
    """

    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "author": {"name": "CK-12 Team"},
          "educationalLevel": "Grade 7",
          "inLanguage": "en-US"
        }
        </script>
      </head>
      <body><p>FlexBook content</p></body>
    </html>
    """

    metadata = ck12_enrichment.parse_flexbook_metadata(html)

    assert metadata == {"author": "CK-12 Team", "grade": "Grade 7", "language": "en"}


def test_extract_pdf_url_prefers_flexbook_pdf() -> None:
    """
    extract_pdf_url should return the strongest PDF candidate from FlexBook HTML.

    Purpose:
        Validate that PDF discovery collects href-like attributes across elements
        and prefers CK-12 export paths that include `/flx/pdf/` with a .pdf
        extension.
    """

    html = """
    <html>
      <body>
        <a href="https://example.com/preview">Preview</a>
        <button data-href="https://flexbooks.ck12.org/flx/pdf/ck-12-algebra-1.pdf">
            Download PDF
        </button>
        <a data-url="https://flexbooks.ck12.org/pdf/alternate.pdf">Alt</a>
      </body>
    </html>
    """

    pdf_url = ck12_enrichment.extract_pdf_url(html)

    assert pdf_url == "https://flexbooks.ck12.org/flx/pdf/ck-12-algebra-1.pdf"


def test_ck12_enrichment_cli_dispatch_happens_after_helper_definitions() -> None:
    """
    `ck12_enrichment` should define helpers before invoking Typer CLI dispatch.

    Purpose:
        Prevent a regression where `python -m ...ck12_enrichment` executes
        `app()` before defining helper functions (like `_read_catalog`,
        `_normalize_text`, etc.). That ordering causes runtime `NameError` when
        the CLI command invokes those helpers.

    Notes:
        This test intentionally checks source ordering rather than executing the
        CLI, to avoid network and filesystem side effects during unit tests.
        We check for `_extract_field` as it's the last helper defined before
        the dispatch block.
    """

    source = Path(ck12_enrichment.__file__).read_text(encoding="utf-8")

    # Find the last helper function definition (the one closest to EOF before dispatch).
    last_helper_index = source.find("def _extract_field")
    assert last_helper_index != -1, "_extract_field helper not found in source"

    dispatch_index = source.rfind('if __name__ == "__main__":')
    assert dispatch_index != -1, "__main__ dispatch block not found in source"

    assert (
        dispatch_index > last_helper_index
    ), "CLI dispatch must happen after all helper definitions"

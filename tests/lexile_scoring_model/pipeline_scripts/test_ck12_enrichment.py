from __future__ import annotations

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    ck12_enrichment,
)


def test_fetch_flexbook_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fetch_flexbook_html should return HTML text when the HTTP call succeeds.

    Purpose:
        Confirm the happy path performs an HTTP GET with the configured timeout
        and returns the response text without alteration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self.status_code = 200
            self.text = "<html>flexbook</html>"

    captured: dict[str, object] = {}

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_enrichment.requests, "get", _fake_get)

    result = ck12_enrichment.fetch_flexbook_html("https://example.com/flexbook")

    assert result == "<html>flexbook</html>"
    assert captured["url"] == "https://example.com/flexbook"
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

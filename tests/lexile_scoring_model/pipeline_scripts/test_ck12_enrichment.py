from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    ck12_enrichment,
)
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    DownloadCandidate,
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


def test_perma_response_yields_revision_download_candidates() -> None:
    """
    extract_revision_download_candidates should emit JSON download candidates.

    Purpose:
        Validate that nested Perma API revision hierarchies produce download
        candidates pointing to the CK-12 revision detail endpoint so downstream
        manifest building can use those URLs.
    """

    perma_response: dict[str, object] = {
        "response": {
            "flexbook": {
                "revisions": [
                    {
                        "children": [
                            {
                                "title": "Chapter 1",
                                "revisions": [
                                    {
                                        "children": [
                                            {"revisionID": 111},
                                            {"revisionID": 222},
                                        ]
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        }
    }

    extractor_name = "extract_revision_download_candidates"
    extractor = cast(
        Callable[[Mapping[str, object]], list[DownloadCandidate]],
        getattr(ck12_enrichment, extractor_name),
    )
    candidates = extractor(perma_response)

    # Validate that extracted revision IDs become distinct revision detail URLs.
    candidate_urls = {candidate.url for candidate in candidates}
    assert candidate_urls == {
        "https://www.ck12.org/flx/get/detail/revision/111?tiny=true",
        "https://www.ck12.org/flx/get/detail/revision/222?tiny=true",
    }
    assert all(candidate.format == "application/json" for candidate in candidates)


def test_fetch_perma_metadata_targets_perma_api_with_required_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    fetch_perma_metadata should call the Perma API with required JSON headers.

    Purpose:
        Ensure enrichment fetches Perma metadata using the canonical artifactType
        and handle while sending the browser-like header set mandated by the spec.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to replace requests.get for isolation.
    """

    class _FakeResponse:
        """Stub response object that mimics requests.Response for success paths."""

        def __init__(self) -> None:
            self._json_data: dict[str, object] = {"response": {"flexbook": {}}}

        def raise_for_status(self) -> None:
            """No-op to simulate a 200 OK status."""
            return None

        def json(self) -> dict[str, object]:
            """Return parsed JSON data."""
            return self._json_data

    captured: dict[str, object] = {}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResponse:
        """Capture request parameters and return a fake response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ck12_enrichment.requests, "get", _fake_get)

    artifact_type = "flexbook"
    handle = "CK-12-Physics-FlexBook-2.0"
    result = ck12_enrichment.fetch_perma_metadata(artifact_type, handle)

    assert result == {"response": {"flexbook": {}}}
    assert (
        captured["url"]
        == f"https://www.ck12.org/flx/get/perma/{artifact_type}/{handle}"
    )

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
    assert timeout_value == ck12_enrichment.REQUEST_TIMEOUT_SECONDS


def test_collect_revision_candidates_reports_skip_when_missing_children() -> None:
    """
    collect_revision_candidates_with_skip_reason should surface a skip reason when no
    revisions exist.

    Purpose:
        Ensure enrichment informs callers/CLI when a Perma payload lacks revisions or
        children by returning an empty candidate list plus a skip reason tuple so the
        CLI can log actionable feedback.
    """

    perma_response: dict[str, object] = {"response": {"flexbook": {"revisions": []}}}

    collector_name = "collect_revision_candidates_with_skip_reason"
    collector = cast(
        Callable[
            [str, Mapping[str, object]],
            tuple[list[DownloadCandidate], tuple[str, str] | None],
        ],
        getattr(ck12_enrichment, collector_name),
    )
    candidates, skip_reason = collector("ck-12-physics-flexbook-2-0", perma_response)

    assert candidates == []
    assert skip_reason == (
        "ck-12-physics-flexbook-2-0",
        "no revisions in perma response",
    )

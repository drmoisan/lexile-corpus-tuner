from __future__ import annotations

from typing import TYPE_CHECKING

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_curation
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
    DownloadCandidate,
)

if TYPE_CHECKING:
    import pytest


def _entry(
    downloads: list[DownloadCandidate], source_id: str = "openstax"
) -> CatalogEntry:
    return CatalogEntry(
        source_id=source_id,
        identifier="identifier",
        title="title",
        creator="creator",
        year="2020",
        language=["eng"],
        license_url="http://license",
        download_candidates=downloads,
    )


def test_has_text_candidate_returns_true_when_text_present() -> None:
    """Entries with a text/plain candidate should pass."""
    entry = _entry([DownloadCandidate(format="text/plain", url="u")])
    assert oer_curation.has_text_candidate(entry) is True


def test_has_text_candidate_returns_false_when_only_pdf() -> None:
    """Entries with only non-text formats should fail the text check."""
    entry = _entry([DownloadCandidate(format="application/pdf", url="u")])
    assert oer_curation.has_text_candidate(entry) is False


def test_has_text_candidate_accepts_openstax_text_among_other_formats() -> None:
    """OpenStax entries remain eligible when a text/plain candidate is present."""
    entry = _entry(
        [
            DownloadCandidate(format="application/pdf", url="pdf"),
            DownloadCandidate(format="text/plain", url="text"),
        ],
        source_id="openstax",
    )
    assert oer_curation.has_text_candidate(entry) is True


def test_has_json_candidate_accepts_application_json() -> None:
    """CK-12 entries with an application/json candidate should pass."""
    entry = _entry(
        [
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/12345?tiny=true",
            )
        ],
        source_id="ck12",
    )
    assert oer_curation.has_json_candidate(entry) is True


def test_has_json_candidate_accepts_revision_path_without_format() -> None:
    """CK-12 entries with revision-detail URLs should pass even without format."""
    entry = _entry(
        [
            DownloadCandidate(
                format="",
                url="https://www.ck12.org/flx/get/detail/revision/67890?tiny=true",
            )
        ],
        source_id="ck12",
    )
    assert oer_curation.has_json_candidate(entry) is True


def test_has_json_candidate_returns_false_for_non_json_candidates() -> None:
    """Non-JSON CK-12 entries should fail the JSON check."""
    entry = _entry(
        [DownloadCandidate(format="application/pdf", url="https://example.com/pdf")],
        source_id="ck12",
    )
    assert oer_curation.has_json_candidate(entry) is False


def test_filter_by_collection_accepts_openstax() -> None:
    """Allowed source should pass collection filter."""
    entry = _entry([], source_id="openstax")
    assert oer_curation.filter_by_collection(entry, ["openstax"]) is True


def test_filter_by_collection_rejects_gutenberg() -> None:
    """Disallowed source should be filtered out."""
    entry = _entry([], source_id="gutenberg")
    assert oer_curation.filter_by_collection(entry, ["openstax"]) is False


def test_curate_entries_returns_included_and_skipped_lists() -> None:
    """curate_entries should separate included vs skipped with reasons."""
    with_text = _entry([DownloadCandidate(format="text/plain", url="u")])
    without_text = _entry([DownloadCandidate(format="application/pdf", url="u2")])
    included, skipped = oer_curation.curate_entries(
        [with_text, without_text], True, ["openstax"]
    )
    assert included == [with_text]
    assert skipped[0][0] == without_text.identifier


def test_curate_entries_logs_skip_reason_for_missing_text() -> None:
    """Skip reason should mention missing text candidate."""
    without_text = _entry([DownloadCandidate(format="application/pdf", url="u2")])
    included, skipped = oer_curation.curate_entries([without_text], True, ["openstax"])
    assert included == []
    assert skipped == [(without_text.identifier, "no text candidate")]


def test_curate_entries_honors_require_pdf_flag() -> None:
    """
    When --require-pdf is set, only PDF candidates are kept and others are skipped.
    """
    pdf_only = _entry(
        [DownloadCandidate(format="application/pdf", url="pdf")], source_id="ck12"
    )
    text_only = _entry(
        [DownloadCandidate(format="text/plain", url="text")], source_id="ck12"
    )
    included, skipped = oer_curation.curate_entries(
        [pdf_only, text_only],
        False,
        ["ck12"],
        require_pdf=True,
    )
    assert included == [pdf_only]
    assert skipped == [(text_only.identifier, "no pdf candidate")]


def test_curate_entries_honors_require_json_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When --require-json is set, only CK-12 entries with revision JSON candidates
    are kept.
    """

    def _always_true(url: str, *, timeout_seconds: float = 0.0) -> bool:
        return True

    with_json = _entry(
        [
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/12345?tiny=true",
            )
        ],
        source_id="ck12",
    )
    without_json = _entry(
        [DownloadCandidate(format="application/pdf", url="https://example.com/pdf")],
        source_id="ck12",
    )
    monkeypatch.setattr(oer_curation, "_is_url_reachable", _always_true)
    included, skipped = oer_curation.curate_entries(
        [with_json, without_json],
        True,
        ["ck12"],
        require_json=True,
    )
    assert included == [with_json]
    assert skipped == [(without_json.identifier, "no json candidate")]


def test_curate_entries_skips_unreachable_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision URLs that return non-200 responses should be skipped."""
    entry = _entry(
        [
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/12345?tiny=true",
            )
        ],
        source_id="ck12",
    )

    def _always_false(url: str, *, timeout_seconds: float = 0.0) -> bool:
        return False

    monkeypatch.setattr(oer_curation, "_is_url_reachable", _always_false)
    included, skipped = oer_curation.curate_entries(
        [entry],
        True,
        ["ck12"],
        require_json=True,
    )
    assert included == []
    assert skipped == [(entry.identifier, "revision url unreachable")]

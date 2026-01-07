from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_curation
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
    DownloadCandidate,
)


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

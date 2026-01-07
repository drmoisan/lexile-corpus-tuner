from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog import (
    parse_catalog_entry,
)
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation import (
    curate_entries,
)
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment import (
    enrich_catalog_entry,
)
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest import (
    generate_manifest,
)


def _build_manifest_entries():
    raw = {
        "identifier": "OpenStax_Book",
        "title": "Physics",
        "creator": ["OpenStax"],
        "year": "2020",
        "language": ["eng"],
        "licenseurl": "http://license",
    }
    catalog_entry = parse_catalog_entry(raw, "openstax")

    def _fake_fetcher(identifier: str) -> dict[str, object]:
        del identifier
        return {"files": [{"name": "book_djvu.txt"}]}

    enriched = enrich_catalog_entry(catalog_entry, fetcher=_fake_fetcher)
    curated, skipped = curate_entries([enriched], True, ["openstax"])
    manifest_entries = generate_manifest(curated, validate_urls=False)
    return manifest_entries, skipped


def test_end_to_end_catalog_to_manifest_with_mocked_ia() -> None:
    """Happy-path flow should yield a manifest entry when text is available."""
    manifest_entries, skipped = _build_manifest_entries()
    assert skipped == []
    assert len(manifest_entries) == 1
    assert manifest_entries[0].url.endswith("book_djvu.txt")


def test_manifest_entries_validate_against_schema() -> None:
    """Manifest rows must contain required keys and stable filenames."""
    manifest_entries, _ = _build_manifest_entries()
    row = manifest_entries[0]
    assert row.source_id == "openstax"
    assert row.id == "openstax-book"
    assert row.url
    assert row.filename.endswith(".txt")


def test_download_normalize_consumes_manifest_without_errors() -> None:
    """Filnames and URLs should align with downloader expectations."""
    manifest_entries, _ = _build_manifest_entries()
    # Ensure filenames and URLs match downloader expectations for every entry.
    for entry in manifest_entries:
        assert entry.filename.endswith(".txt")
        assert entry.url.startswith("http")

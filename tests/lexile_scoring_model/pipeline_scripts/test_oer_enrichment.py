from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_enrichment
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
)


def test_extract_text_candidates_filters_djvu_txt_files() -> None:
    """_djvu.txt files should be preferred and included."""
    files = [
        {"name": "book_djvu.txt"},
        {"name": "book.txt"},
        {"name": "book.pdf"},
    ]
    candidates = oer_enrichment.extract_text_candidates("identifier", files)
    assert candidates[0].url.endswith("book_djvu.txt")


def test_extract_text_candidates_excludes_pdf_files() -> None:
    """PDF files must not be included as text candidates."""
    files = [{"name": "book.pdf"}]
    candidates = oer_enrichment.extract_text_candidates("identifier", files)
    assert candidates == []


def test_extract_text_candidates_includes_text_plain_format() -> None:
    """Plain .txt files should be included when present."""
    files = [{"name": "chapter1.txt"}]
    candidates = oer_enrichment.extract_text_candidates("identifier", files)
    assert candidates[0].format == "text/plain"


def test_enrich_catalog_entry_appends_download_candidates() -> None:
    """enrich_catalog_entry should attach discovered download candidates."""
    entry = CatalogEntry(
        source_id="openstax",
        identifier="openstax-book",
        title="Physics",
        creator="OpenStax",
        year="2020",
        language=["eng"],
        license_url="http://license",
    )

    def _fake_fetcher(identifier: str) -> dict[str, object]:
        del identifier
        return {"files": [{"name": "file_djvu.txt"}]}

    enriched = oer_enrichment.enrich_catalog_entry(entry, fetcher=_fake_fetcher)
    assert len(enriched.download_candidates) == 1
    assert enriched.download_candidates[0].url.endswith("file_djvu.txt")

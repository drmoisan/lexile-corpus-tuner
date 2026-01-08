from __future__ import annotations

import json
from typing import TYPE_CHECKING

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_catalog
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_build_ia_query_for_openstax_includes_collection_filter() -> None:
    """OpenStax query should target OpenStax collection keywords."""
    query = oer_catalog.build_ia_query("openstax")
    assert "openstax" in query.lower()
    assert "textbook" in query.lower()


def test_build_ia_query_for_ck12_includes_collection_filter() -> None:
    """CK-12 query should target CK-12 keywords."""
    query = oer_catalog.build_ia_query("ck12")
    assert "ck12" in query.lower()
    assert "flexbook" in query.lower()


def test_parse_catalog_entry_extracts_all_required_fields() -> None:
    """Parsing should populate all CatalogEntry fields when present."""
    raw = {
        "identifier": "openstax-book",
        "title": "Physics",
        "creator": ["OpenStax"],
        "year": "2020",
        "language": ["eng"],
        "licenseurl": "http://license",
    }
    entry = oer_catalog.parse_catalog_entry(raw, "openstax")
    assert entry.identifier == "openstax-book"
    assert entry.source_id == "openstax"
    assert entry.title == "Physics"
    assert entry.creator == "OpenStax"
    assert entry.year == "2020"
    assert entry.language == ["eng"]
    assert entry.license_url == "http://license"


def test_parse_catalog_entry_handles_missing_optional_fields() -> None:
    """Missing optional fields should be converted to None or empty lists."""
    raw = {"identifier": "ck12-algebra"}
    entry = oer_catalog.parse_catalog_entry(raw, "ck12")
    assert entry.title is None
    assert entry.creator is None
    assert entry.year is None
    assert entry.language == []


def test_write_catalog_jsonl_creates_valid_jsonl_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """write_catalog_jsonl should produce JSONL lines that parse cleanly."""
    output_file = tmp_path / "catalog.jsonl"

    entries = [
        CatalogEntry(
            source_id="openstax",
            identifier="openstax-book",
            title="Physics",
            creator="OpenStax",
            year="2020",
            language=["eng"],
            license_url="http://license",
            download_candidates=[],
        )
    ]
    oer_catalog.write_catalog_jsonl(entries, output_file)

    # Read back the JSONL and validate
    lines: list[str] = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["identifier"] == "openstax-book"
    assert parsed["download_candidates"] == []

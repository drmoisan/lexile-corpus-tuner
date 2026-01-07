from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_catalog
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write_catalog_jsonl should produce JSONL lines that parse cleanly."""
    buffer = StringIO()
    fake_path = Path("/fake/catalog.jsonl")

    def _fake_open(
        self: Path, mode: str, encoding: str | None = None
    ) -> AbstractContextManager[StringIO]:
        # Provide a context manager over the shared buffer.
        del mode, encoding
        buffer.seek(0)
        buffer.truncate(0)

        class _Ctx:
            def __enter__(self) -> StringIO:
                return buffer

            def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
                return None

        return _Ctx()

    monkeypatch.setattr(Path, "open", _fake_open)
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
    oer_catalog.write_catalog_jsonl(entries, fake_path)
    lines: list[str] = []
    # Collect non-empty JSONL lines emitted by the writer.
    for line in buffer.getvalue().splitlines():
        if line.strip():
            lines.append(line)
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["identifier"] == "openstax-book"
    assert parsed["download_candidates"] == []

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_ui
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
    DownloadCandidate,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pytest
    from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest import (
        ManifestEntry,
    )


def _catalog_line(identifier: str) -> str:
    payload = {
        "source_id": "openstax",
        "identifier": identifier,
        "title": "title",
        "creator": "creator",
        "year": "2020",
        "language": ["eng"],
        "license_url": "http://license",
        "download_candidates": [
            {"format": "text/plain", "url": "http://example.com/file.txt", "size": None}
        ],
    }
    return json.dumps(payload)


def test_load_catalog_files_reads_all_jsonl_in_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_catalog_files should parse every JSONL file returned by glob."""

    class _FakeFile:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def read_text(self, encoding: str = "utf-8") -> str:  # type: ignore[override]
            del encoding
            return self._payload

    fake_dir = Path("/fake")
    file_one = _FakeFile(_catalog_line("id1") + "\n")
    file_two = _FakeFile(_catalog_line("id2") + "\n")

    def _fake_glob(self: Path, pattern: str) -> Iterator[_FakeFile]:  # type: ignore[override]
        del pattern
        yield file_one
        yield file_two

    monkeypatch.setattr(Path, "glob", _fake_glob)
    entries = oer_ui.load_catalog_files(fake_dir)
    identifiers: set[str] = set()
    # Collect identifiers from every parsed entry to confirm both files loaded.
    for entry in entries:
        identifiers.add(entry.identifier)
    assert identifiers == {"id1", "id2"}


def test_catalog_viewmodel_toggle_selection_updates_state() -> None:
    """Toggling a selection should flip the stored state."""
    entries = [CatalogEntry("openstax", "id", "t", "c", "2020", ["eng"], None)]
    viewmodel = oer_ui.CatalogViewModel(entries)
    assert viewmodel.get_selected_entries() == []
    viewmodel.toggle_selection(0)
    assert viewmodel.get_selected_entries()[0].identifier == "id"


def test_catalog_viewmodel_get_selected_entries_returns_only_selected() -> None:
    """Only selected entries should be returned."""
    entries = [
        CatalogEntry("openstax", "id1", "t", "c", "2020", ["eng"], None),
        CatalogEntry("openstax", "id2", "t", "c", "2020", ["eng"], None),
    ]
    viewmodel = oer_ui.CatalogViewModel(entries)
    viewmodel.toggle_selection(1)
    selected = viewmodel.get_selected_entries()
    assert len(selected) == 1
    assert selected[0].identifier == "id2"


def test_export_manifest_calls_manifest_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export_manifest should invoke manifest generation and writing helpers."""
    called: dict[str, object] = {}

    def _fake_generate(entries: Iterable[CatalogEntry], validate_urls: bool = False):
        called["validate_urls"] = validate_urls
        called["count"] = len(list(entries))
        return [
            DownloadCandidate(format="text/plain", url="http://example.com/file.txt")
        ]

    def _fake_write(entries: list[DownloadCandidate], output_path: Path):  # type: ignore[override]
        called["output"] = output_path.as_posix()
        called["written"] = entries

    monkeypatch.setattr(oer_ui, "generate_manifest", _fake_generate)
    monkeypatch.setattr(oer_ui, "write_manifest_json", _fake_write)

    entries = [CatalogEntry("openstax", "id1", "t", "c", "2020", ["eng"], None)]
    oer_ui.export_manifest(entries, Path("/fake/out.json"))

    assert called["validate_urls"] is False
    assert called["count"] == 1
    assert called["output"] == "/fake/out.json"


def test_export_manifest_produces_json_for_ck12_and_txt_for_openstax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mixed-source selection produces correct file extensions.

    Scenario gate: CK-12 entries emit `.json` filenames while OpenStax entries
    emit `.txt` filenames. This validates that the UI export_manifest function,
    which delegates to generate_manifest, correctly routes extensions by source.
    """
    captured_entries: list[ManifestEntry] = []

    def _capture_write(entries: list[ManifestEntry], output_path: Path) -> None:
        del output_path
        captured_entries.extend(entries)

    monkeypatch.setattr(oer_ui, "write_manifest_json", _capture_write)

    # Build mixed-source entries: one CK-12 (JSON candidate) and one OpenStax (text).
    ck12_entry = CatalogEntry(
        source_id="ck12",
        identifier="ck-12-physics-flexbook-2-0",
        title="CK-12 Physics FlexBook 2.0",
        creator="CK-12 Foundation",
        year="2024",
        language=["eng"],
        license_url="https://www.ck12.org/terms",
        download_candidates=[
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/8384007?tiny=true",
                size=None,
            )
        ],
    )
    openstax_entry = CatalogEntry(
        source_id="openstax",
        identifier="CollegeAlgebraCorequisite2e",
        title="College Algebra Corequisite Support 2e",
        creator="OpenStax",
        year="2023",
        language=["eng"],
        license_url="http://creativecommons.org/licenses/by/4.0/",
        download_candidates=[
            DownloadCandidate(
                format="text/plain",
                url="https://archive.org/download/CollegeAlgebraCorequisite2e/CollegeAlgebraCorequisite2e_djvu.txt",
                size=None,
            )
        ],
    )

    oer_ui.export_manifest([ck12_entry, openstax_entry], Path("/fake/manifest.json"))

    # Validate extension mapping is correct.
    assert len(captured_entries) == 2
    # Find entries by source and validate their extensions.
    ck12_manifest = next(e for e in captured_entries if e.source_id == "ck12")
    openstax_manifest = next(e for e in captured_entries if e.source_id == "openstax")
    assert ck12_manifest.filename.endswith(
        ".json"
    ), f"CK-12 should emit .json, got {ck12_manifest.filename}"
    assert openstax_manifest.filename.endswith(
        ".txt"
    ), f"OpenStax should emit .txt, got {openstax_manifest.filename}"

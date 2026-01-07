"""
Tkinter-based curation UI for OER catalogs.

Purpose:
    Provide a lightweight visual review tool so curators can load catalog
    entries, toggle inclusion, apply simple filters, and export a manifest
    compatible with the downloader/normalizer.

Usage:
    - `load_catalog_files` reads catalog JSONL files into CatalogEntry objects.
    - `CatalogViewModel` tracks selection state for the loaded entries.
    - `create_catalog_table` renders a checkbox list bound to the view model.
    - `create_filter_panel` offers text filters that invoke a caller-supplied
      callback when applied.
    - `export_manifest` emits a manifest using manifest generation helpers.
    - CLI `curate_oer_ui` launches a minimal Tkinter window for manual curation.

Flow:
    The CLI loads catalogs from `data/meta/catalogs`, initializes a view model,
    renders the table + filter panel, and allows the user to export a manifest
    to `data/meta/oer_sources.json` via the UI button.

Invariants / Constraints:
    - Only catalogs with JSONL rows containing the expected CatalogEntry fields
      are supported.
    - Filenames in manifests must remain `.txt` to satisfy the normalizer.

Side Effects:
    - CLI performs file reads and manifest writes when the user exports.
"""

from __future__ import annotations

import json
import tkinter as tk
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer

from .oer_manifest import generate_manifest, write_manifest_json
from .oer_models import CatalogEntry, DownloadCandidate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _to_optional_str(value: object) -> str | None:
    """Convert a possibly-null value to a string or None."""
    if value is None:
        return None
    return str(value)


def _to_str_list(value: object | list[object] | None) -> list[str]:
    """Convert a scalar or list value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        items = cast(list[object], value)
        return [str(item) for item in items]
    return [str(value)]


app = typer.Typer(help="Visual curation UI for OER catalogs.")


def _candidate_from_mapping(mapping: Mapping[str, object | None]) -> DownloadCandidate:
    """Convert a raw mapping to a DownloadCandidate with safe defaults."""
    format_raw = mapping.get("format")
    url_raw = mapping.get("url")
    size_raw = mapping.get("size")
    return DownloadCandidate(
        format=str(format_raw) if format_raw is not None else "",
        url=str(url_raw) if url_raw is not None else "",
        size=size_raw if isinstance(size_raw, int) else None,
    )


def _catalog_from_line(raw: Mapping[str, object | list[object] | None]) -> CatalogEntry:
    """Restore a CatalogEntry from a JSON-decoded mapping."""
    candidates: list[DownloadCandidate] = []
    candidates_value = raw.get("download_candidates")
    candidate_mappings: list[Mapping[str, object | None]] = []
    if isinstance(candidates_value, list):
        # Rebuild download candidates from serialized dictionaries.
        candidate_list = cast(list[Mapping[str, object | None]], candidates_value)
        for candidate in candidate_list:
            candidate_mappings.append(candidate)
    for candidate_mapping in candidate_mappings:
        candidates.append(_candidate_from_mapping(candidate_mapping))

    language_value: object | list[object] | None = raw.get("language")
    languages = _to_str_list(language_value)

    source_id = _to_optional_str(raw.get("source_id"))
    identifier_raw = raw.get("identifier")
    identifier = "" if identifier_raw is None else str(identifier_raw)
    title = _to_optional_str(raw.get("title"))
    creator_raw = raw.get("creator")
    creator = None
    if isinstance(creator_raw, list):
        creator_items = cast(list[object], creator_raw)
        if creator_items:
            creator = str(creator_items[0])
    elif creator_raw is not None:
        creator = str(creator_raw)
    year = _to_optional_str(raw.get("year"))
    license_url = _to_optional_str(raw.get("license_url"))

    return CatalogEntry(
        source_id=source_id,
        identifier=identifier,
        title=title,
        creator=creator,
        year=year,
        language=languages,
        license_url=license_url,
        download_candidates=candidates,
    )


def load_catalog_files(catalog_dir: Path) -> list[CatalogEntry]:
    """
    Load all JSONL catalog files in a directory into CatalogEntry objects.

    Args:
        catalog_dir: Directory containing catalog JSONL files.

    Returns:
        List of CatalogEntry objects aggregated from all files.
    """
    entries: list[CatalogEntry] = []
    # Walk every catalog JSONL and accumulate entries for the UI to present.
    for path in sorted(catalog_dir.glob("*.jsonl"), key=lambda p: str(p)):
        # Decode each JSONL row to rebuild CatalogEntry objects.
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = cast(Mapping[str, object | list[object] | None], json.loads(line))
            entries.append(_catalog_from_line(raw))
    return entries


class CatalogViewModel:
    """
    Track catalog entries and selection state for the UI.

    Responsibilities:
        - Expose the current entries for rendering.
        - Toggle selection state per entry.
        - Provide the selected subset for export.
    """

    def __init__(self, entries: Iterable[CatalogEntry]):
        self._entries: list[CatalogEntry] = list(entries)
        self._selected: list[bool] = [False] * len(self._entries)

    def get_entries(self) -> list[CatalogEntry]:
        """Return the catalog entries in display order."""
        return list(self._entries)

    def toggle_selection(self, index: int) -> None:
        """Flip selection state for the entry at the given index."""
        if index < 0 or index >= len(self._selected):
            raise IndexError("Selection index out of range")
        self._selected[index] = not self._selected[index]

    def get_selected_entries(self) -> list[CatalogEntry]:
        """Return only the entries currently marked as selected."""
        selected: list[CatalogEntry] = []
        # Collect only entries whose selection flag is true.
        for entry, is_selected in zip(self._entries, self._selected, strict=False):
            if is_selected:
                selected.append(entry)
        return selected

    def is_selected(self, index: int) -> bool:
        """Expose selection state for UI binding."""
        if index < 0 or index >= len(self._selected):
            return False
        return self._selected[index]


def create_catalog_table(parent: tk.Misc, viewmodel: CatalogViewModel) -> tk.Frame:
    """
    Build a simple checkbox table bound to the view model.

    Args:
        parent: Parent widget.
        viewmodel: CatalogViewModel supplying entries and selection state.

    Returns:
        Frame containing checkboxes and labels.
    """
    frame = tk.Frame(parent)
    header = tk.Frame(frame)
    tk.Label(header, text="Select").grid(row=0, column=0, padx=4, sticky="w")
    tk.Label(header, text="Identifier").grid(row=0, column=1, padx=4, sticky="w")
    tk.Label(header, text="Title").grid(row=0, column=2, padx=4, sticky="w")
    header.pack(fill="x")

    # Render each entry with a checkbox so the curator can toggle inclusion.
    for idx, entry in enumerate(viewmodel.get_entries()):
        row = tk.Frame(frame)
        var = tk.BooleanVar(value=viewmodel.is_selected(idx))

        def _toggle(i: int, state_var: tk.BooleanVar) -> Callable[[], None]:
            return lambda: _toggle_helper(viewmodel, i, state_var)

        tk.Checkbutton(row, variable=var, command=_toggle(idx, var)).grid(
            row=0, column=0, padx=4, sticky="w"
        )
        tk.Label(row, text=entry.identifier).grid(row=0, column=1, padx=4, sticky="w")
        tk.Label(row, text=entry.title or "(untitled)").grid(
            row=0, column=2, padx=4, sticky="w"
        )
        row.pack(fill="x", pady=1)
    return frame


def _toggle_helper(
    viewmodel: CatalogViewModel, index: int, state_var: tk.BooleanVar
) -> None:
    """Toggle selection in the view model and sync the checkbox state."""
    viewmodel.toggle_selection(index)
    state_var.set(viewmodel.is_selected(index))


def create_filter_panel(
    parent: tk.Misc, on_filter: Callable[[str, str, str], None]
) -> tk.Frame:
    """
    Create a filter panel with subject, grade, and language entries.

    Args:
        parent: Parent widget.
        on_filter: Callback invoked with (subject, grade, language) values when
            the Apply button is pressed.

    Returns:
        Frame containing filter controls.
    """
    frame = tk.Frame(parent)
    tk.Label(frame, text="Subject").grid(row=0, column=0, padx=4, pady=2, sticky="w")
    subject_entry = tk.Entry(frame)
    subject_entry.grid(row=0, column=1, padx=4, pady=2)

    tk.Label(frame, text="Grade").grid(row=1, column=0, padx=4, pady=2, sticky="w")
    grade_entry = tk.Entry(frame)
    grade_entry.grid(row=1, column=1, padx=4, pady=2)

    tk.Label(frame, text="Language").grid(row=2, column=0, padx=4, pady=2, sticky="w")
    language_entry = tk.Entry(frame)
    language_entry.grid(row=2, column=1, padx=4, pady=2)

    def _on_apply() -> None:
        on_filter(subject_entry.get(), grade_entry.get(), language_entry.get())

    tk.Button(frame, text="Apply Filters", command=_on_apply).grid(
        row=3, column=0, columnspan=2, pady=4
    )
    return frame


def export_manifest(entries: list[CatalogEntry], output_path: Path) -> None:
    """
    Generate and write a manifest from selected entries.

    Args:
        entries: Curated/selected catalog entries.
        output_path: Destination path for the manifest JSON.
    """
    manifest_entries = generate_manifest(entries, validate_urls=False)
    write_manifest_json(manifest_entries, output_path)


@app.command()
def curate_oer_ui() -> None:
    """Launch the Tkinter UI for manual curation."""
    catalog_dir = Path("data/meta/catalogs")
    entries = load_catalog_files(catalog_dir)
    viewmodel = CatalogViewModel(entries)

    root = tk.Tk()
    root.title("OER Curation")

    table = create_catalog_table(root, viewmodel)
    table.pack(fill="both", expand=True, padx=8, pady=8)

    def _apply_filters(subject: str, grade: str, language: str) -> None:
        # Intentional no-op stub; real filtering can be added later without
        # altering the manifest generation contract.
        _ = (subject, grade, language)

    filter_panel = create_filter_panel(root, _apply_filters)
    filter_panel.pack(fill="x", padx=8, pady=4)

    def _export() -> None:
        selected = viewmodel.get_selected_entries()
        export_manifest(selected, Path("data/meta/oer_sources.json"))
        typer.echo(f"Exported manifest with {len(selected)} entries")

    tk.Button(root, text="Export Manifest", command=_export).pack(pady=8)
    root.mainloop()


if __name__ == "__main__":
    app()

"""
Enrich catalog entries with download candidates using IA Metadata API.

Purpose:
    Given a CatalogEntry from search results, discover downloadable text
    derivatives and attach them as DownloadCandidate objects.

Flow:
    - fetch_ia_metadata retrieves the file list for an IA identifier.
    - extract_text_candidates filters for text-friendly files (_djvu.txt first).
    - enrich_catalog_entry appends candidates to the entry.
    - CLI `enrich_oer_catalog` reads a catalog JSONL, enriches, and writes back.

Constraints:
    - Prefer `_djvu.txt`, then other `.txt` files; ignore PDFs/EPUBs here.
    - Network calls should be mocked in tests; this module isolates HTTP.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)  # noqa: TCH003 - runtime for Typer type hints
from pathlib import Path  # noqa: TCH003 - Path required at runtime for Typer and IO
from typing import Any, cast

import typer

from .oer_models import CatalogEntry, DownloadCandidate

IA_METADATA_ENDPOINT = "https://archive.org/metadata/{identifier}"

app = typer.Typer(help="Enrich catalog entries with IA download candidates.")


def fetch_ia_metadata(
    identifier: str,
) -> dict[str, object]:  # pragma: no cover - network helper exercised indirectly
    """Fetch IA metadata for a single identifier."""
    url = IA_METADATA_ENDPOINT.format(identifier=identifier)
    req = urllib.request.Request(url)  # noqa: S310 - IA HTTPS endpoint is expected
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        payload = resp.read()
    return cast(dict[str, object], json.loads(payload.decode("utf-8", errors="ignore")))


def extract_text_candidates(
    identifier: str, files: Sequence[Mapping[str, object]]
) -> list[DownloadCandidate]:
    """
    Extract text-friendly download candidates from an IA metadata file list.

    Preference order:
    1) Filenames ending with `_djvu.txt`
    2) Other `.txt` files
    """
    candidates: list[DownloadCandidate] = []
    # Gather filenames once to apply preference ordering deterministically.
    # Collect candidate filenames once for deterministic ordering.
    name_candidates = (file_obj.get("name") for file_obj in files)
    names: list[str] = [name for name in name_candidates if isinstance(name, str)]
    preferred = [n for n in names if n.endswith("_djvu.txt")]
    secondary = [n for n in names if n.endswith(".txt") and not n.endswith("_djvu.txt")]
    ordered = preferred + secondary
    # Build download candidates for each discovered text file.
    for name in ordered:
        url = f"https://archive.org/download/{identifier}/{name}"
        candidates.append(DownloadCandidate(format="text/plain", url=url, size=None))
    return candidates


def enrich_catalog_entry(
    entry: CatalogEntry,
    fetcher: Callable[[str], dict[str, object]] | None = None,
) -> CatalogEntry:
    """
    Enrich a CatalogEntry with download candidates from IA metadata.
    """
    http: Callable[[str], dict[str, object]] = fetcher or fetch_ia_metadata
    metadata = http(entry.identifier)
    files_raw = cast(Sequence[Mapping[str, object]], metadata.get("files") or [])
    candidates = extract_text_candidates(entry.identifier, files_raw)
    return CatalogEntry(
        source_id=entry.source_id,
        identifier=entry.identifier,
        title=entry.title,
        creator=entry.creator,
        year=entry.year,
        language=entry.language,
        license_url=entry.license_url,
        download_candidates=candidates,
    )


def _read_catalog(path: Path) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    # Parse each JSONL line into CatalogEntry instances.
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = cast(dict[str, Any], json.loads(line))
        raw_candidates = cast(
            list[dict[str, object]], raw.get("download_candidates", [])
        )
        candidates = [_candidate_from_mapping(c) for c in raw_candidates]
        entries.append(
            CatalogEntry(
                source_id=raw.get("source_id"),
                identifier=raw["identifier"],
                title=raw.get("title"),
                creator=raw.get("creator"),
                year=raw.get("year"),
                language=raw.get("language") or [],
                license_url=raw.get("license_url"),
                download_candidates=candidates,
            )
        )
    return entries


def _candidate_from_mapping(candidate: dict[str, object]) -> DownloadCandidate:
    """Convert a raw mapping into a DownloadCandidate with safe defaults."""
    format_raw = candidate.get("format")
    url_raw = candidate.get("url")
    size_raw = candidate.get("size")
    return DownloadCandidate(
        format=str(format_raw) if format_raw is not None else "",
        url=str(url_raw) if url_raw is not None else "",
        size=size_raw if isinstance(size_raw, int) else None,
    )


def _write_catalog(path: Path, entries: list[CatalogEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        # Persist entries line-by-line to keep files streamable.
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "source_id": entry.source_id,
                        "identifier": entry.identifier,
                        "title": entry.title,
                        "creator": entry.creator,
                        "year": entry.year,
                        "language": entry.language,
                        "license_url": entry.license_url,
                        "download_candidates": [
                            c.__dict__ for c in entry.download_candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


@app.command()
def enrich_oer_catalog(  # pragma: no cover - CLI wrapper
    catalog_file: Path = typer.Option(..., exists=True, readable=True),  # noqa: B008
    output: Path = typer.Option(  # noqa: B008
        ..., help="Path to write enriched catalog JSONL"
    ),
) -> None:
    """
    CLI: read a catalog JSONL, enrich entries, and write output.
    """
    entries = _read_catalog(catalog_file)
    enriched: list[CatalogEntry] = []
    # Enrich each entry one-by-one to avoid partial mutation if a request fails.
    # Enrich sequentially so failures can be logged without aborting everything.
    for entry in entries:
        try:
            enriched.append(enrich_catalog_entry(entry))
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Skipping {entry.identifier}: {exc}", err=True)
    _write_catalog(output, enriched)
    typer.echo(f"Wrote enriched catalog to {output} ({len(enriched)} entries)")


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

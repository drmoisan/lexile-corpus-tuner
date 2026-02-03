"""
CK-12 FlexBook enrichment with Perma/Revision traversal.

Purpose:
    Provide a typed surface for fetching CK-12 Perma metadata, extracting
    revision IDs, and attaching revision-detail download candidates to catalog
    entries while preserving testability and stable identifiers.

Usage:
    `enrich_ck12_catalog` reads catalog JSONL rows (containing canonical CK-12
    handles), calls the Perma API for each entry, converts nested revisions to
    download candidates, and writes an enriched JSONL artifact. Legacy helpers
    for HTML/PDF parsing remain for backward compatibility and existing tests.

Flow:
    1) Read catalog entries and canonical handles.
    2) Fetch Perma metadata with browser-like headers.
    3) Traverse revisions/children to collect revision IDs.
    4) Emit revision-detail download candidates into enriched catalog rows.

Invariants / Constraints:
    - Slugs derived from catalog entries must remain stable.
    - Enrichment should not mutate unrelated catalog fields.
    - Network and filesystem interactions must remain mockable for tests.

Side Effects:
    Performs HTTP requests to CK-12 APIs and filesystem writes when invoked via CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import typer

from .ck12_catalog import write_catalog_jsonl
from .ck12_enrichment_core import (
    PERMA_REQUEST_HEADERS,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT_SECONDS,
    collect_revision_candidates_with_skip_reason,
    extract_pdf_url,
    extract_revision_download_candidates,
    fetch_flexbook_html,
    fetch_perma_metadata,
    normalize_language_code,
    normalize_text,
    parse_flexbook_metadata,
)
from .oer_models import CatalogEntry, DownloadCandidate

__all__ = [
    "PERMA_REQUEST_HEADERS",
    "REQUEST_HEADERS",
    "REQUEST_TIMEOUT_SECONDS",
    "collect_revision_candidates_with_skip_reason",
    "extract_pdf_url",
    "extract_revision_download_candidates",
    "fetch_flexbook_html",
    "fetch_perma_metadata",
    "parse_flexbook_metadata",
]

app = typer.Typer(
    help="Enrich CK-12 catalog entries with CK-12 Perma revision metadata."
)


def enrich_entry_logic(
    entry: CatalogEntry, metadata: dict[str, str | None], pdf_url: str | None
) -> CatalogEntry:
    """
    Merge parsed metadata and PDF link into a catalog entry.

    Purpose:
        Produce an updated `CatalogEntry` with enrichment applied while
        preserving identifier stability and existing catalog fields.

    Args:
        entry (CatalogEntry): Base catalog entry from the CK-12 catalog stage.
        metadata (dict[str, str | None]): Parsed metadata fields such as author,
            grade, and language.
        pdf_url (str | None): PDF download link when available.

    Returns:
        CatalogEntry: New entry instance containing merged enrichment data.

    Raises:
        ValueError: If the provided PDF URL is not an absolute HTTP(S) URL.

    Side Effects:
        None. Pure function returning a new catalog entry instance.
    """
    normalized_pdf_url = normalize_text(pdf_url)
    if pdf_url is not None:
        if normalized_pdf_url is None:
            raise ValueError("PDF URL must be non-empty when provided")
        parsed_pdf = urlparse(normalized_pdf_url)
        if parsed_pdf.scheme not in {"http", "https"} or not parsed_pdf.netloc:
            raise ValueError("PDF URL must be an absolute HTTP(S) URL")

    language_candidate = normalize_language_code(metadata.get("language"))
    merged_languages: list[str] = list(entry.language)
    # Preserve existing language ordering while appending new metadata when absent.
    if language_candidate and language_candidate not in merged_languages:
        merged_languages.append(language_candidate)

    candidates: list[DownloadCandidate] = list(entry.download_candidates)
    # Attach a PDF download candidate when provided, avoiding duplicate URLs.
    if normalized_pdf_url is not None:
        pdf_candidate = DownloadCandidate(
            format="application/pdf", url=normalized_pdf_url, size=None
        )
        if not any(candidate.url == pdf_candidate.url for candidate in candidates):
            candidates.append(pdf_candidate)

    return CatalogEntry(
        source_id=entry.source_id,
        identifier=entry.identifier,
        title=entry.title,
        creator=metadata.get("author") or entry.creator,
        year=entry.year,
        language=merged_languages,
        license_url=entry.license_url,
        download_candidates=candidates,
    )


@app.command()
def enrich_ck12_catalog(
    catalog_file: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/meta/catalogs/ck12_catalog.jsonl"),
        help="Path to the CK-12 catalog JSONL file to enrich.",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/meta/catalogs/ck12_enriched.jsonl"),
        help="Path where the enriched catalog JSONL will be written.",
    ),
) -> None:
    """
    CLI entry point for enriching CK-12 catalog entries with Perma revision metadata.

    Purpose:
        Fetch Perma API payloads for each catalog row, extract revision IDs, and
        attach revision-detail download candidates so downstream curation can
        require JSON assets instead of PDFs.

    Args:
        catalog_file (Path): Input catalog JSONL produced by the catalog step.
        output (Path): Destination for the enriched JSONL artifact.

    Returns:
        None

    Side Effects:
        Performs HTTP requests to CK-12 APIs and filesystem writes.
    """
    entries = _read_catalog(catalog_file)
    enriched_entries: list[CatalogEntry] = []
    skip_reasons: list[tuple[str, str]] = []

    # Iterate deterministically so failures on one row do not corrupt others.
    for entry in entries:
        artifact_type = entry.artifact_type or "flexbook"
        handle = entry.handle

        # Skip entries missing canonical handles because Perma API cannot resolve them.
        if handle is None:
            skip_reasons.append((entry.identifier, "missing handle"))
            typer.echo(f"Skipping {entry.identifier} due to missing handle", err=True)
            continue

        try:
            perma_payload = fetch_perma_metadata(artifact_type, handle)
            candidates, skip_reason = collect_revision_candidates_with_skip_reason(
                entry.identifier, perma_payload
            )
            enriched_entries.append(
                CatalogEntry(
                    source_id=entry.source_id,
                    identifier=entry.identifier,
                    title=entry.title,
                    creator=entry.creator,
                    year=entry.year,
                    language=entry.language,
                    license_url=entry.license_url,
                    download_candidates=candidates,
                    artifact_type=artifact_type,
                    handle=handle,
                    artifact_id=entry.artifact_id,
                )
            )
            if skip_reason:
                skip_reasons.append(skip_reason)
        except Exception as exc:  # noqa: BLE001 - CLI top-level enrichment loop
            typer.echo(
                f"Skipping {entry.identifier} due to enrichment error: {exc}", err=True
            )

    write_catalog_jsonl(enriched_entries, output)
    for identifier, reason in skip_reasons:
        typer.echo(f"{identifier}: {reason}", err=True)
    typer.echo(f"Wrote {len(enriched_entries)} enriched CK-12 entries to {output}")


def _read_catalog(path: Path) -> list[CatalogEntry]:
    """
    Read a catalog JSONL file into CatalogEntry objects.

    Purpose:
        Provide a typed loader for CK-12 catalog rows so enrichment can operate
        on consistent value objects while preserving download candidates.

    Args:
        path (Path): Path to the catalog JSONL file.

    Returns:
        list[CatalogEntry]: Parsed catalog entries in file order.
    """
    entries: list[CatalogEntry] = []

    # Iterate JSONL lines to reconstruct CatalogEntry objects with CK-12 metadata.
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = cast("dict[str, Any]", json.loads(line))
        raw_candidates = cast(
            "list[dict[str, object]]", raw.get("download_candidates", [])
        )
        candidates = [
            _candidate_from_mapping(candidate) for candidate in raw_candidates
        ]
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
                artifact_type=cast("str | None", raw.get("artifact_type")),
                handle=cast("str | None", raw.get("handle")),
                artifact_id=_parse_artifact_id(raw.get("artifact_id")),
            )
        )

    return entries


def _candidate_from_mapping(candidate: dict[str, object]) -> DownloadCandidate:
    """
    Convert a raw mapping into a DownloadCandidate with safe defaults.

    Purpose:
        Normalize persisted download candidate dictionaries back into typed
        objects without assuming field presence.

    Args:
        candidate (dict[str, object]): Raw dictionary from JSONL.

    Returns:
        DownloadCandidate: Rehydrated download candidate instance.
    """
    format_raw = candidate.get("format")
    url_raw = candidate.get("url")
    size_raw = candidate.get("size")
    return DownloadCandidate(
        format=str(format_raw) if format_raw is not None else "",
        url=str(url_raw) if url_raw is not None else "",
        size=size_raw if isinstance(size_raw, int) else None,
    )


def _parse_artifact_id(value: object) -> int | None:
    """
    Normalize artifact_id values from JSON to integers when possible.

    Purpose:
        Preserve CK-12 artifact IDs for downstream debugging or logging while
        tolerating missing or non-numeric values.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

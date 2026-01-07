"""
Manifest generation for curated OER entries.

Purpose:
    Convert curated CatalogEntry records into manifest entries consumable by the
    downloader and normalizer. Optional URL validation is supported via HTTP
    HEAD requests.

Flow:
    - Read curated catalog JSONL files.
    - Build ManifestEntry objects using stable slugs derived from identifiers.
    - Optionally validate URLs (HTTP 200 + text/* Content-Type).
    - Write `data/meta/oer_sources.json` (or caller-provided path).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import (  # noqa: TCH003 - type hints consumed at runtime
    Iterable,
    Mapping,
)
from pathlib import Path
from typing import Any, cast

import typer

from .oer_models import (
    CatalogEntry,
    DownloadCandidate,
    ManifestEntry,
    generate_stable_slug,
)

app = typer.Typer(help="Generate OER manifest from curated catalogs.")


def _to_candidate(
    candidate: Mapping[str, object] | DownloadCandidate,
) -> DownloadCandidate:
    """Normalize raw candidate dictionaries into DownloadCandidate objects."""
    if isinstance(candidate, DownloadCandidate):
        return candidate
    format_raw = candidate.get("format")
    url_raw = candidate.get("url")
    size_raw = candidate.get("size")
    format_value = str(format_raw) if format_raw is not None else ""
    url_value = str(url_raw) if url_raw is not None else ""
    size_value = size_raw if isinstance(size_raw, int) else None
    return DownloadCandidate(
        format=format_value,
        url=url_value,
        size=size_value,
    )


def build_manifest_entry(
    catalog_entry: CatalogEntry, candidate: DownloadCandidate
) -> ManifestEntry:
    """
    Construct a ManifestEntry from a curated catalog entry and chosen candidate.

    Purpose:
        Produce the manifest row required by downstream download/normalize
        stages, enforcing stable slug IDs and `.txt` filenames.

    Args:
        catalog_entry: Curated entry representing a single IA item.
        candidate: Selected download candidate for the text derivative.

    Returns:
        ManifestEntry: Immutable manifest row.
    """
    slug = generate_stable_slug(catalog_entry.identifier)
    filename = f"{slug}.txt"
    return ManifestEntry(
        source_id=catalog_entry.source_id or "oer",
        id=slug,
        url=candidate.url,
        filename=filename,
    )


def validate_url(url: str) -> tuple[bool, int | None, str | None]:
    """
    Perform a HEAD request to verify reachability and content type.

    Args:
        url: Target URL to validate.

    Returns:
        Tuple of (is_valid, status_code, content_type). is_valid is False when
        the request fails, returns a non-200 status, or when the content type
        does not begin with text/*.
    """
    req = urllib.request.Request(  # noqa: S310 - IA HTTPS endpoint is expected
        url, method="HEAD"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            status = resp.getcode()
            content_type = resp.headers.get("Content-Type")
    except Exception:
        return False, None, None
    if status != 200:
        return False, status, content_type
    if content_type and not content_type.lower().startswith("text"):
        return False, status, content_type
    return True, status, content_type


def generate_manifest(
    curated_entries: Iterable[CatalogEntry],
    validate_urls: bool = False,
) -> list[ManifestEntry]:
    """
    Build manifest entries from curated catalog entries.

    Purpose:
        Transform curated entries into manifest rows and optionally validate
        URLs to catch dead links before downstream processing.

    Args:
        curated_entries: Iterable of curated CatalogEntry objects.
        validate_urls: When True, perform HTTP HEAD checks before inclusion.

    Returns:
        List of ManifestEntry objects ready for serialization.
    """
    manifest_entries: list[ManifestEntry] = []
    # Evaluate each curated entry and pick the first text candidate.
    for entry in curated_entries:
        chosen = next(
            (
                candidate
                for candidate in entry.download_candidates
                if candidate.format.startswith("text/")
            ),
            None,
        )
        if not chosen:
            continue
        manifest_entry = build_manifest_entry(entry, chosen)
        if validate_urls:
            ok, status, content_type = validate_url(manifest_entry.url)
            if not ok:
                message = (
                    f"Skipping {manifest_entry.id}: validation failed "
                    f"(status={status}, content_type={content_type})"
                )
                typer.echo(message, err=True)
                continue
        manifest_entries.append(manifest_entry)
    return manifest_entries


def write_manifest_json(entries: list[ManifestEntry], output_path: Path) -> None:
    """
    Write manifest entries to JSON under the expected schema.

    Args:
        entries: Manifest rows to persist.
        output_path: Destination JSON path (parent dirs created automatically).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sources": [entry.__dict__ for entry in entries]}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_curated(catalog_dir: Path) -> list[CatalogEntry]:
    """Read curated JSONL files from a directory into CatalogEntry objects."""
    entries: list[CatalogEntry] = []
    catalogs = sorted(catalog_dir.glob("*_curated.jsonl"))
    # Read each curated file to assemble the full set.
    for path in catalogs:
        # Decode JSONL lines into CatalogEntry instances for this curated file.
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = cast(dict[str, Any], json.loads(line))
            candidates_raw = cast(
                list[Mapping[str, object]], raw.get("download_candidates", [])
            )
            # Normalize raw candidates so downstream consumers receive typed objects.
            candidates: list[DownloadCandidate] = []
            for candidate in candidates_raw:
                candidates.append(_to_candidate(candidate))
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


@app.command()
def generate_oer_manifest(  # pragma: no cover - CLI wrapper
    catalog_dir: Path = typer.Option(..., exists=True, file_okay=False),  # noqa: B008
    out: Path = typer.Option(  # noqa: B008
        Path("data/meta/oer_sources.json"), help="Output manifest path"
    ),
    validate_urls: bool = typer.Option(  # noqa: B008
        False, help="Validate URLs via HTTP HEAD"
    ),
) -> None:
    """
    CLI entrypoint to generate the OER manifest from curated catalogs.

    Args:
        catalog_dir: Directory containing *_curated.jsonl files.
        out: Destination for the manifest JSON payload.
        validate_urls: When True, perform HEAD checks before inclusion.

    Side Effects:
        - Reads curated JSONL files from disk.
        - Performs optional network requests for validation.
        - Writes manifest JSON to disk, creating parent dirs if needed.
    """
    curated_entries = _read_curated(catalog_dir)
    manifest_entries = generate_manifest(curated_entries, validate_urls=validate_urls)
    write_manifest_json(manifest_entries, out)
    typer.echo(f"Wrote manifest with {len(manifest_entries)} entries to {out}")


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

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

import hashlib
import json
import re
import urllib.request
from collections.abc import (  # noqa: TCH003 - type hints consumed at runtime
    Iterable,
    Mapping,
)
from concurrent.futures import ThreadPoolExecutor
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


_CK12_REVISION_ID_PATTERN = re.compile(r"/flx/get/detail/revision/(?P<revision_id>\d+)")


def _ck12_revision_suffix(url: str) -> str:
    """
    Produce a stable CK-12 suffix used to prevent manifest ID collisions.

    Purpose:
        CK-12 curated catalog entries represent book-level identifiers, but the
        Perma API can surface many revision-detail URLs (lessons/sections).
        The manifest must keep each revision distinct so the downloader does
        not overwrite artifacts.

    Args:
        url (str): CK-12 revision-detail URL.

    Returns:
        str: Suffix beginning with "--" that uniquely identifies the revision.

    Raises:
        None.

    Side Effects:
        None.
    """
    match = _CK12_REVISION_ID_PATTERN.search(url)
    if match:
        return f"--rev-{match.group('revision_id')}"

    # Fall back to a short URL hash so filenames remain deterministic even
    # if the URL format changes.
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"--urlhash-{digest}"


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
        stages, enforcing stable slug IDs and filename extensions aligned to
        source expectations (CK-12 revision JSON uses `.json`; PDFs stay `.pdf`;
        everything else defaults to `.txt`).

    Args:
        catalog_entry: Curated entry representing a single IA item.
        candidate: Selected download candidate for the text derivative.

    Returns:
        ManifestEntry: Immutable manifest row.
    """
    slug = generate_stable_slug(catalog_entry.identifier)
    source = (catalog_entry.source_id or "").lower()
    format_lower = candidate.format.lower()
    # Branch extensions by source + candidate format:
    # - CK-12 entries are revision JSON responses and must emit `.json`.
    # - PDF candidates stay `.pdf`.
    # - All other derivatives default to `.txt`.
    if source == "ck12":
        extension = ".json"
    elif format_lower.startswith("application/pdf"):
        extension = ".pdf"
    else:
        extension = ".txt"

    # CK-12 manifest entries must be per-revision to avoid collisions: a single
    # book identifier can map to many revision-detail URLs.
    if source == "ck12":
        manifest_id = f"{slug}{_ck12_revision_suffix(candidate.url)}"
    else:
        manifest_id = slug

    filename = f"{manifest_id}{extension}"
    return ManifestEntry(
        source_id=catalog_entry.source_id or "oer",
        id=manifest_id,
        url=candidate.url,
        filename=filename,
    )


def validate_url(
    url: str, allowed_content_types: list[str] | None = None
) -> tuple[bool, int | None, str | None]:
    """
    Perform a HEAD request to verify reachability and content type.

    Args:
        url: Target URL to validate.
        allowed_content_types: Acceptable Content-Type prefixes (case-insensitive).
            Defaults to ["text"] to permit text/* responses. When provided, the
            HEAD response must start with one of the allowed prefixes.

    Returns:
        Tuple of (is_valid, status_code, content_type). is_valid is False when
        the request fails, returns a non-200 status, or when the content type
        does not begin with an allowed prefix.
    """
    # Normalize allowed content-type prefixes for case-insensitive comparison.
    content_type_prefixes = [
        value.lower() for value in (allowed_content_types or ["text"])
    ]
    # User-Agent required: CK-12 CloudFront blocks requests without it.
    req = urllib.request.Request(  # noqa: S310 - trusted HTTPS endpoint
        url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"},
    )
    try:
        # Keep validation latency bounded so manifest generation stays usable
        # even when a subset of URLs time out.
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            status = resp.getcode()
            content_type = resp.headers.get("Content-Type")
    except Exception:
        return False, None, None
    if status != 200:
        return False, status, content_type
    if content_type and not any(
        content_type.lower().startswith(prefix) for prefix in content_type_prefixes
    ):
        return False, status, content_type
    return True, status, content_type


def _iter_manifest_candidates(
    curated_entries: Iterable[CatalogEntry],
) -> Iterable[tuple[ManifestEntry, list[str]]]:
    """
    Yield manifest candidates and their allowed content types.

    Purpose:
        Centralize candidate selection rules so manifest generation and
        validation share identical logic.

    Args:
        curated_entries: Curated CatalogEntry objects (typically from JSONL).

    Yields:
        Tuples of (ManifestEntry, allowed_content_types) in deterministic order.

    Side Effects:
        None.
    """

    # Evaluate each curated entry and pick the format expected for that source
    # (JSON for CK-12 revision payloads, text for OpenStax, PDF where explicitly
    # advertised).
    for entry in curated_entries:
        # Route candidate selection by source expectations:
        # - CK-12 uses revision JSON (`application/json` or revision-detail URLs).
        # - Other sources remain text-first using text/* derivatives.
        if (entry.source_id or "").lower() == "ck12":
            # CK-12 book entries typically expand to multiple revision-detail
            # candidates (lessons/sections). Emit one manifest row per candidate.
            chosen_candidates = [
                candidate
                for candidate in entry.download_candidates
                if candidate.format.startswith("application/json")
                or "/flx/get/detail/revision/" in candidate.url
            ]
            allowed_content_types = ["application/json"]
        else:
            chosen_candidates: list[DownloadCandidate] = []
            chosen = next(
                (
                    candidate
                    for candidate in entry.download_candidates
                    if candidate.format.startswith("text/")
                ),
                None,
            )
            if chosen is not None:
                chosen_candidates.append(chosen)
            allowed_content_types = ["text"]

        if not chosen_candidates:
            continue

        # Deduplicate candidates by URL to avoid emitting duplicate rows when
        # upstream catalogs include repeated revision-detail links.
        seen_urls: set[str] = set()
        for candidate in chosen_candidates:
            if candidate.url in seen_urls:
                continue
            seen_urls.add(candidate.url)

            yield build_manifest_entry(entry, candidate), allowed_content_types


def _validate_manifest_candidate(
    candidate: tuple[ManifestEntry, list[str]],
) -> tuple[ManifestEntry, bool, int | None, str | None]:
    """
    Validate a single manifest candidate.

    Purpose:
        Provide a small, pickle-free wrapper that can be executed in a thread
        pool while keeping logging and ordering decisions in the caller.

    Args:
        candidate: (ManifestEntry, allowed_content_types) pair.

    Returns:
        (ManifestEntry, is_valid, status_code, content_type)
    """

    manifest_entry, allowed_content_types = candidate
    ok, status, content_type = validate_url(
        manifest_entry.url,
        allowed_content_types=allowed_content_types,
    )
    return manifest_entry, ok, status, content_type


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
    manifest_candidates = _iter_manifest_candidates(curated_entries)

    if not validate_urls:
        # Emit all candidates without network validation.
        return [manifest_entry for manifest_entry, _allowed in manifest_candidates]

    manifest_entries: list[ManifestEntry] = []

    # Validate URLs concurrently to keep end-to-end runtime reasonable when
    # HEAD requests are slow, while preserving deterministic manifest order.
    #
    # PERF: This is network-bound, so a larger worker count is typically safe.
    # We intentionally avoid materializing a full list of futures so large
    # manifests (e.g., CK-12 revisions) do not incur unnecessary overhead.
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Preserve output order by using executor.map(), which yields results in
        # the same order as the input iterable.
        for manifest_entry, ok, status, content_type in executor.map(
            _validate_manifest_candidate,
            manifest_candidates,
        ):
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

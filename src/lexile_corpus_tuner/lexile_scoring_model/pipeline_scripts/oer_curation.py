"""
Curation logic for OER catalog entries.

Purpose:
    Filter catalog entries down to those with viable text downloads and allowed
    sources, while recording skip reasons for auditability.

Flow:
    - has_text_candidate: ensure at least one text/plain download exists.
    - has_pdf_candidate: ensure at least one application/pdf download exists.
    - filter_by_collection: enforce allowed source_ids.
    - curate_entries: produce included list and skipped list with reasons.
    - CLI `curate_oer_catalog` applies the filters to catalog files and writes
      curated JSONL outputs plus a skip log.

Constraints:
    - Current pipeline only supports openstax/ck12 with text/plain downloads.
    - Skip reasons must be explicit for downstream diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import requests
import typer

from .oer_models import CatalogEntry, DownloadCandidate

ALLOWED_DEFAULT = ["openstax", "ck12"]
REVISION_HEAD_TIMEOUT_SECONDS = 10.0
app = typer.Typer(help="Curate OER catalog entries and log skip reasons.")


def has_text_candidate(entry: CatalogEntry) -> bool:
    """Return True if any download candidate is text/plain."""
    return any(c.format.startswith("text/") for c in entry.download_candidates)


def has_pdf_candidate(entry: CatalogEntry) -> bool:
    """
    Return True if any download candidate represents a PDF file.

    Purpose:
        Support CK-12 curation steps that require downloadable PDFs by detecting
        either an explicit application/pdf MIME type or a PDF-looking URL when
        format metadata is absent.

    Args:
        entry (CatalogEntry): Catalog entry whose download candidates are checked.

    Returns:
        bool: True when at least one candidate appears to be a PDF.
    """
    # Accept declared MIME types first, then fall back to URL suffix for resilience.
    for candidate in entry.download_candidates:
        format_value = candidate.format.lower()
        if format_value.startswith("application/pdf"):
            return True
        if not format_value and candidate.url.lower().endswith(".pdf"):
            return True
    return False


def has_json_candidate(entry: CatalogEntry) -> bool:
    """
    Return True when a candidate looks like a CK-12 revision JSON payload.

    Purpose:
        Detect CK-12 revision detail downloads for JSON-required curation flows
        by inspecting both declared MIME types and known CK-12 revision paths.

    Args:
        entry (CatalogEntry): Catalog entry whose download candidates are checked.

    Returns:
        bool: True when a revision JSON candidate is present.

    Side Effects:
        None.
    """
    for candidate in entry.download_candidates:
        # Accept explicit JSON declarations and known CK-12 revision endpoints.
        format_value = candidate.format.lower()
        if format_value.startswith("application/json"):
            return True
        if "/flx/get/detail/revision/" in candidate.url.lower():
            return True
    return False


# HTTP headers required for CK-12 API HEAD requests to avoid 403 responses.
CK12_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.ck12.org/",
    "Origin": "https://www.ck12.org",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def _is_url_reachable(
    url: str, *, timeout_seconds: float = REVISION_HEAD_TIMEOUT_SECONDS
) -> bool:
    """
    Return True when the URL responds with HTTP 200 using a HEAD request.

    Purpose:
        Prevent inclusion of CK-12 revision entries whose URLs are already
        unreachable during curation, avoiding downstream download failures.

    Args:
        url (str): Candidate URL to probe.
        timeout_seconds (float): Timeout in seconds for the HEAD request.

    Returns:
        bool: True when the URL returns HTTP 200; False on non-200 or exceptions.

    Side Effects:
        Issues a network HEAD request with a short timeout.
    """
    # CK-12 API requires browser-like headers to avoid 403 Forbidden responses.
    headers = CK12_REQUEST_HEADERS if "ck12.org" in url.lower() else None
    try:
        response = requests.head(
            url, allow_redirects=True, timeout=timeout_seconds, headers=headers
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def _has_reachable_revision_candidate(entry: CatalogEntry) -> bool:
    """
    Return True when a CK-12 revision candidate responds with HTTP 200.

    Purpose:
        Enforce reachability checks for revision JSON URLs so that unreachable
        CK-12 entries are skipped early in the curation process.

    Args:
        entry (CatalogEntry): Catalog entry containing download candidates.

    Returns:
        bool: True when at least one revision-like candidate is reachable.

    Side Effects:
        Performs HEAD requests for revision-like candidates.
    """
    # Probe each revision-like candidate and accept the first reachable one.
    for candidate in entry.download_candidates:
        format_value = candidate.format.lower()
        if not (
            format_value.startswith("application/json")
            or "/flx/get/detail/revision/" in candidate.url.lower()
        ):
            continue
        if _is_url_reachable(candidate.url):
            return True
    return False


def filter_by_collection(entry: CatalogEntry, allowed: list[str]) -> bool:
    """Return True when the entry source_id is in the allowed list."""
    return entry.source_id in allowed


def curate_entries(
    entries: list[CatalogEntry],
    require_text: bool,
    allowed_sources: list[str],
    *,
    require_pdf: bool = False,
    require_json: bool = False,
) -> tuple[list[CatalogEntry], list[tuple[str, str]]]:
    """
    Split entries into included and skipped sets with reasons.

    Purpose:
        Evaluate catalog entries against format and source constraints while
        returning deterministic skip reasons for downstream diagnostics.

    Args:
        entries (list[CatalogEntry]): Catalog rows to evaluate.
        require_text (bool): When True, keep only entries with text/plain candidates.
        allowed_sources (list[str]): source_id values permitted for inclusion.
        require_pdf (bool): When True, keep only entries with PDF candidates.
        require_json (bool): When True, keep only entries with revision JSON candidates.

    Returns:
        tuple[list[CatalogEntry], list[tuple[str, str]]]: Included entries and skipped
        identifier/reason pairs.

    Side Effects:
        None.
    """
    included: list[CatalogEntry] = []
    skipped: list[tuple[str, str]] = []
    effective_require_text = require_text and not require_pdf and not require_json
    # Evaluate each entry sequentially to capture the first failing reason.
    for entry in entries:
        # Enforce requested format requirements before checking source membership.
        if require_pdf and not has_pdf_candidate(entry):
            skipped.append((entry.identifier, "no pdf candidate"))
            continue
        if require_json:
            if not has_json_candidate(entry):
                skipped.append((entry.identifier, "no json candidate"))
                continue
            if not _has_reachable_revision_candidate(entry):
                skipped.append((entry.identifier, "revision url unreachable"))
                continue
        if effective_require_text and not has_text_candidate(entry):
            skipped.append((entry.identifier, "no text candidate"))
            continue
        if not filter_by_collection(entry, allowed_sources):
            skipped.append((entry.identifier, "source not allowed"))
            continue
        included.append(entry)
    return included, skipped


def _read_catalog(path: Path) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    # Decode JSONL to restore CatalogEntry objects.
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
    """Normalize raw candidate dictionaries into DownloadCandidate objects."""
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
        # Emit JSONL for included entries.
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
                            {
                                "format": candidate.format,
                                "url": candidate.url,
                                "size": candidate.size,
                            }
                            for candidate in entry.download_candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_skips(path: Path, skipped: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        # Record identifier plus reason for every skip.
        for identifier, reason in skipped:
            handle.write(
                json.dumps({"identifier": identifier, "reason": reason}) + "\n"
            )


@app.command()
def curate_oer_catalog(  # pragma: no cover - CLI wrapper
    # noqa: B008 - Typer option defaults are runtime configuration
    catalog_dir: Path = typer.Option(..., exists=True, file_okay=False),  # noqa: B008
    require_text: bool = typer.Option(  # noqa: B008
        True, help="Require at least one text/plain candidate"
    ),
    require_pdf: bool = typer.Option(  # noqa: B008
        False, help="Require at least one application/pdf candidate"
    ),
    require_json: bool = typer.Option(  # noqa: B008
        False, help="Require at least one CK-12 revision JSON candidate"
    ),
    sources: str = typer.Option(
        "openstax,ck12", help="Comma-separated allowed sources"
    ),
    out_dir: Path = typer.Option(  # noqa: B008
        Path("data/meta/catalogs"), help="Output directory for curated files"
    ),
) -> None:
    """CLI to curate catalogs in a directory and emit curated files + skip logs."""
    allowed = [s.strip().lower() for s in sources.split(",") if s.strip()]
    # Prefer enriched catalogs when available (they contain full metadata + PDFs).
    base_catalogs = sorted(catalog_dir.glob("*_catalog.jsonl"))
    catalogs: list[Path] = []
    for base_path in base_catalogs:
        enriched_path = base_path.parent / base_path.name.replace(
            "_catalog", "_enriched"
        )
        # Use enriched version when it exists, otherwise fall back to base catalog.
        catalogs.append(enriched_path if enriched_path.exists() else base_path)
    # When PDF or JSON is explicitly required, skip the text/plain check to support
    # CK-12 flows.
    effective_require_text = require_text and not require_pdf and not require_json
    # Curate each catalog independently so outputs remain per-source.
    for catalog_path in catalogs:
        entries = _read_catalog(catalog_path)
        included, skipped = curate_entries(
            entries,
            effective_require_text,
            allowed,
            require_pdf=require_pdf,
            require_json=require_json,
        )
        # Derive curated/skip names from the source prefix
        # (before _catalog or _enriched).
        # Input: ck12_catalog.jsonl or ck12_enriched.jsonl
        # Output: ck12_curated.jsonl
        if "_enriched.jsonl" in catalog_path.name:
            base_name = catalog_path.name.replace("_enriched.jsonl", "")
        elif "_catalog.jsonl" in catalog_path.name:
            base_name = catalog_path.name.replace("_catalog.jsonl", "")
        else:
            base_name = catalog_path.stem
        curated_path = out_dir / f"{base_name}_curated.jsonl"
        skips_path = out_dir / f"{base_name}_skips.jsonl"
        _write_catalog(curated_path, included)
        _write_skips(skips_path, skipped)
        typer.echo(
            f"Curated {catalog_path.name}: {len(included)} kept, "
            f"{len(skipped)} skipped -> {curated_path}"
        )


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

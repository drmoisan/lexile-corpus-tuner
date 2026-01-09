"""
CK-12 FlexBook enrichment scaffold with typed interfaces for metadata and PDF discovery.

Purpose:
    Provide a strongly-typed surface for fetching FlexBook pages, extracting
    metadata, and merging enrichment details into catalog entries while keeping
    network and parsing concerns isolated for testability.

Usage:
    Future implementations will call `fetch_flexbook_html` to retrieve the page,
    parse enrichment fields via `parse_flexbook_metadata`, locate a PDF link with
    `extract_pdf_url`, and merge results using `enrich_entry_logic`. The
    `enrich_ck12_catalog` CLI will orchestrate these steps for pipeline
    operators.

Flow:
    1) Fetch FlexBook HTML for each catalog row.
    2) Parse author/grade/language/license fields and locate PDF download link.
    3) Merge enrichment into `CatalogEntry` and persist to an enriched JSONL file.

Invariants / Constraints:
    - Slugs derived from catalog entries must remain stable.
    - Enrichment should not mutate unrelated catalog fields.
    - Network and filesystem interactions must remain mockable for tests.

Side Effects:
    Implementation will perform HTTP requests, HTML parsing, and filesystem I/O;
    this stub performs no I/O.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import typer
from bs4 import BeautifulSoup

from .ck12_catalog import write_catalog_jsonl
from .oer_models import CatalogEntry, DownloadCandidate

REQUEST_TIMEOUT_SECONDS = 30
FLEXBOOK_BASE_URL = "https://flexbooks.ck12.org/cbook/"

# HTTP headers to mimic a real browser and avoid 403 Forbidden (Issue #73)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9," "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

app = typer.Typer(
    help="Enrich CK-12 catalog entries with metadata and PDF download links."
)


def fetch_flexbook_html(url: str) -> str:
    """
    Retrieve a CK-12 FlexBook page for enrichment.

    Purpose:
        Centralize HTTP retrieval so timeouts, retries, and error handling can
        be applied consistently across enrichment calls.
        Includes browser-like headers to avoid 403 Forbidden responses.

    Args:
        url (str): Fully-qualified FlexBook URL to fetch.

    Returns:
        str: Raw HTML content of the FlexBook page.

    Raises:
        RuntimeError: When the request fails or returns a non-success status.

    Side Effects:
        Implementation will issue an HTTP GET request with a timeout and
        browser headers.
    """
    try:
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to fetch FlexBook HTML from {url}") from exc

    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"FlexBook request to {url} returned status {response.status_code}"
        )

    return response.text


def parse_flexbook_metadata(html: str) -> dict[str, str | None]:
    """
    Extract enrichment fields from FlexBook HTML.

    Purpose:
        Parse author, grade, language, and related metadata in a single pass so
        downstream merging can rely on a consistent dictionary shape. This
        routine favors structured JSON-LD blocks, then falls back to meta tags
        and simple text heuristics.

    Args:
        html (str): HTML document retrieved from the FlexBook page.

    Returns:
        dict[str, str | None]: Mapping of metadata keys to parsed values, using
            None when a field is unavailable.

    Side Effects:
        None. Pure transformation of provided HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    metadata: dict[str, str | None] = {"author": None, "grade": None, "language": None}

    # Favor language hints from the document root to seed defaults.
    language_hint = _normalize_language_code(
        _get_first_str(soup.html.get("lang")) if soup.html is not None else None
    )
    if language_hint:
        metadata["language"] = language_hint

    # Inspect JSON-LD blocks first because they often include normalized fields.
    for script_tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        script_content = script_tag.string or script_tag.get_text()
        if not script_content:
            continue

        try:
            parsed_json = json.loads(script_content)
        except json.JSONDecodeError:
            continue

        json_objects = cast(
            "list[object]",
            parsed_json if isinstance(parsed_json, list) else [parsed_json],
        )

        # Process each JSON object independently to harvest structured metadata.
        for node in json_objects:
            if not isinstance(node, dict):
                continue

            typed_node = cast("dict[str, object]", node)

            if metadata["author"] is None:
                metadata["author"] = _extract_author_from_ldjson(typed_node)

            if metadata["grade"] is None:
                metadata["grade"] = _normalize_text(
                    _extract_field(typed_node, ["educationalLevel", "grade"])
                )

            if metadata["language"] is None:
                metadata["language"] = _normalize_language_code(
                    _extract_field(typed_node, ["inLanguage", "language"])
                )

    # Scan meta tags for remaining gaps such as author or locale hints.
    for meta_tag in soup.find_all("meta"):
        raw_name = meta_tag.get("name") or meta_tag.get("property")
        name_attr = (_get_first_str(raw_name) or "").lower()

        content = _normalize_text(_get_first_str(meta_tag.get("content")))
        if not content:
            continue

        # Pull author from common meta definitions if still missing.
        if metadata["author"] is None and name_attr in {
            "author",
            "dc.creator",
            "book:author",
        }:
            metadata["author"] = content

        # Capture grade metadata that may be exposed as a dedicated meta tag.
        if metadata["grade"] is None and name_attr in {"grade", "book:grade"}:
            metadata["grade"] = content

        # Prefer explicit language declarations before regex-based fallbacks.
        if metadata["language"] is None and name_attr in {"language", "og:locale"}:
            metadata["language"] = _normalize_language_code(content)

        if all(value is not None for value in metadata.values()):
            break

    # As a final fallback, scan visible text for a grade pattern.
    if metadata["grade"] is None:
        page_text = soup.get_text(" ", strip=True)
        grade_match = re.search(
            r"\bgrade\s*[:\-]?\s*([Kk0-9][\w\s\-–]*)", page_text, flags=re.IGNORECASE
        )
        if grade_match:
            metadata["grade"] = _normalize_text(grade_match.group(1))

    return metadata


def extract_pdf_url(html: str) -> str | None:
    """
    Locate a PDF download link within the FlexBook HTML.

    Purpose:
        Identify the best PDF candidate so enrichment can attach a direct
        download URL when available.

    Args:
        html (str): HTML document retrieved from the FlexBook page.

    Returns:
        str | None: Direct PDF URL if found; otherwise None.

    Raises:
        ValueError: If a discovered PDF link is malformed.

    Side Effects:
        None. Pure transformation of provided HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    pdf_candidates: list[str] = []
    seen: set[str] = set()

    # Scan anchor/button elements for href-like attributes pointing to PDF exports.
    for tag in soup.find_all(["a", "button"]):
        for attr in ("href", "data-href", "data-url"):
            raw_value = _get_first_str(tag.get(attr))
            candidate = _normalize_text(raw_value)
            if candidate is None:
                continue

            lowered = candidate.lower()
            if ".pdf" not in lowered and "/pdf/" not in lowered:
                continue

            if candidate not in seen:
                pdf_candidates.append(candidate)
                seen.add(candidate)

    if not pdf_candidates:
        return None

    def _score_candidate(url: str) -> int:
        lowered = url.lower()
        score = 0
        if "/flx/pdf/" in lowered:
            score += 3
        if lowered.endswith(".pdf"):
            score += 2
        if "/pdf/" in lowered:
            score += 1
        return score

    # Prefer URLs that match CK-12 export paths and explicit .pdf extensions.
    best_candidate = max(pdf_candidates, key=_score_candidate)

    parsed = urlparse(best_candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"PDF link is not an absolute URL: {best_candidate}")

    if not parsed.path.lower().endswith(".pdf"):
        raise ValueError(f"PDF link does not point to a PDF file: {best_candidate}")

    return best_candidate


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
    normalized_pdf_url = _normalize_text(pdf_url)
    if pdf_url is not None:
        if normalized_pdf_url is None:
            raise ValueError("PDF URL must be non-empty when provided")
        parsed_pdf = urlparse(normalized_pdf_url)
        if parsed_pdf.scheme not in {"http", "https"} or not parsed_pdf.netloc:
            raise ValueError("PDF URL must be an absolute HTTP(S) URL")

    language_candidate = _normalize_language_code(metadata.get("language"))
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
    CLI entry point for enriching CK-12 catalog entries with metadata and PDFs.

    Purpose:
        Orchestrate fetch, parse, and enrichment steps so operators can generate
        an enriched CK-12 catalog with a single command.

    Args:
        catalog_file (Path): Input catalog JSONL produced by the catalog step.
        output (Path): Destination for the enriched JSONL artifact.

    Returns:
        None

    Side Effects:
        Implementation will perform HTTP requests, parsing, and filesystem writes.
    """
    entries = _read_catalog(catalog_file)
    enriched_entries: list[CatalogEntry] = []

    # Process entries sequentially so a failure on one row does not corrupt others.
    for entry in entries:
        try:
            flexbook_url = _build_flexbook_url(entry.identifier)
            html = fetch_flexbook_html(flexbook_url)
            metadata = parse_flexbook_metadata(html)
            pdf_url = extract_pdf_url(html)
            enriched_entries.append(enrich_entry_logic(entry, metadata, pdf_url))
        except Exception as exc:  # noqa: BLE001 - CLI top-level enrichment loop
            typer.echo(
                f"Skipping {entry.identifier} due to enrichment error: {exc}", err=True
            )

    write_catalog_jsonl(enriched_entries, output)
    typer.echo(f"Wrote {len(enriched_entries)} enriched CK-12 entries to {output}")


def _build_flexbook_url(identifier: str) -> str:
    """
    Construct the FlexBook URL from a catalog identifier slug.

    Purpose:
        Catalog entries store only the stable slug; this helper rebuilds the
        absolute FlexBook URL needed to fetch enrichment HTML.

    Args:
        identifier (str): Stable slug derived from the FlexBook path segment.

    Returns:
        str: Fully-qualified FlexBook URL ending with a trailing slash.

    Raises:
        ValueError: If the identifier is empty after normalization.
    """
    normalized_slug = _normalize_text(identifier)
    if normalized_slug is None:
        raise ValueError("Catalog entry identifier cannot be empty")
    return f"{FLEXBOOK_BASE_URL}{normalized_slug}/"


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

    # Iterate JSONL lines to reconstruct CatalogEntry objects.
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


def _get_first_str(value: str | list[str] | None) -> str | None:
    """
    Extract the first string from a potentially list-valued attribute.

    Purpose:
        Handle BeautifulSoup returning list[str] for multi-valued attributes
        (like 'class') or single strings for others, flattening to a scalar.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_text(value: str | None) -> str | None:
    """
    Collapse whitespace and return None when the value is empty after trimming.

    Purpose:
        Provide consistent cleanup for metadata fragments that may include extra
        spacing or line breaks in HTML or JSON-LD payloads.

    Args:
        value (str | None): Raw text to normalize.

    Returns:
        str | None: Cleaned text when present; otherwise None.
    """
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _normalize_language_code(value: str | None) -> str | None:
    """
    Normalize language codes by lowercasing and collapsing locale suffixes.

    Purpose:
        Standardize language hints that may appear as BCP 47 tags (e.g., en-US)
        or shorthand names so downstream stages can compare codes reliably.

    Args:
        value (str | None): Raw language string from HTML or JSON-LD.

    Returns:
        str | None: Normalized lowercase language code without locale suffix, or
            None when the input is empty.
    """
    cleaned = _normalize_text(value)
    if cleaned is None:
        return None
    normalized = cleaned.lower().replace("_", "-")
    # Keep the primary language subtag to align with catalog language lists.
    return normalized.split("-")[0] if normalized else None


def _extract_author_from_ldjson(node: dict[str, object]) -> str | None:
    """
    Pull an author name from a JSON-LD dictionary.

    Purpose:
        Handle the variety of shapes CK-12 may emit for author metadata (string,
        dict with name, or list of authors) without leaking parsing logic into
        the main metadata routine.

    Args:
        node (dict[str, object]): JSON-LD object potentially containing author.

    Returns:
        str | None: Author name if present; otherwise None.
    """
    author_field = node.get("author")
    if isinstance(author_field, str):
        return _normalize_text(author_field)

    if isinstance(author_field, dict):
        author_data = cast("dict[str, object]", author_field)
        return _normalize_text(cast("str | None", author_data.get("name")))

    if isinstance(author_field, list):
        # Prefer the first author entry when multiple are provided.
        for author_candidate in cast("list[object]", author_field):
            if isinstance(author_candidate, dict):
                candidate_data = cast("dict[str, object]", author_candidate)
                name_value = _normalize_text(
                    cast("str | None", candidate_data.get("name"))
                )
                if name_value:
                    return name_value
            elif isinstance(author_candidate, str):
                name_value = _normalize_text(author_candidate)
                if name_value:
                    return name_value

    return None


def _extract_field(node: dict[str, object], keys: list[str]) -> str | None:
    """
    Extract a string field from a JSON-LD dictionary using the provided keys.

    Purpose:
        Centralize tolerant extraction for fields that may be strings, lists, or
        nested dictionaries while keeping the main parser concise.

    Args:
        node (dict[str, object]): JSON-LD object to inspect.
        keys (list[str]): Candidate keys to try in order.

    Returns:
        str | None: First usable string value found; otherwise None.
    """
    for key in keys:
        if key not in node:
            continue

        value = node[key]
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            # Grab the first non-empty string entry.
            for item in cast("list[object]", value):
                if isinstance(item, str):
                    normalized_item = _normalize_text(item)
                    if normalized_item:
                        return normalized_item

        if isinstance(value, dict) and "name" in value:
            typed_val = cast("dict[str, object]", value)
            nested_value = typed_val.get("name")
            if isinstance(nested_value, str):
                return nested_value

    return None


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

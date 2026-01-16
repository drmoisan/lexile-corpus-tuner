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
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import typer
from bs4 import BeautifulSoup

from .ck12_catalog import write_catalog_jsonl
from .oer_models import CatalogEntry, DownloadCandidate

REQUEST_TIMEOUT_SECONDS = 30

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

# HTTP headers for CK-12 JSON APIs (Browse/Perma/Revision detail).
PERMA_REQUEST_HEADERS = {
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

app = typer.Typer(
    help="Enrich CK-12 catalog entries with CK-12 Perma revision metadata."
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


def fetch_perma_metadata(artifact_type: str, handle: str) -> dict[str, object]:
    """
    Retrieve CK-12 Perma API metadata for the specified artifact.

    Purpose:
        Fetch the Perma endpoint using canonical artifact type and handle while
        sending the browser-like JSON headers required for anonymous access.

    Args:
        artifact_type (str): CK-12 artifact type such as "flexbook".
        handle (str): Canonical CK-12 handle from the catalog entry.

    Returns:
        dict[str, object]: Parsed JSON payload from the Perma API.

    Raises:
        RuntimeError: If the request fails, returns a non-success status, or the
            payload is not a JSON object.

    Side Effects:
        Issues an HTTP GET request with timeouts and required headers.
    """
    perma_url = f"https://www.ck12.org/flx/get/perma/{artifact_type}/{handle}"
    try:
        response = requests.get(
            perma_url,
            headers=PERMA_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        json_data = response.json()
        if not isinstance(json_data, dict):
            raise RuntimeError(
                "Unexpected CK-12 Perma payload type "
                f"{type(json_data)} from {perma_url}"
            )
        return cast(dict[str, object], json_data)
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timed out while fetching CK-12 Perma metadata from {perma_url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON received from CK-12 Perma at {perma_url}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid JSON received from CK-12 Perma at {perma_url}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch CK-12 Perma metadata from {perma_url}"
        ) from exc


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


def extract_revision_download_candidates(
    perma_response: Mapping[str, object],
) -> list[DownloadCandidate]:
    """
    Convert Perma API revision hierarchies into download candidates.

    Purpose:
        Traverse the nested revisions/children structure returned by the Perma API
        and emit deterministic download candidates pointing at the CK-12 revision
        detail endpoint for each discovered revision identifier.

    Args:
        perma_response (Mapping[str, object]): Parsed JSON payload from the Perma
            API, expected to include a `response` object containing the artifact
            with nested `revisions`/`children` nodes.

    Returns:
        list[DownloadCandidate]: Download candidates targeting revision detail
            URLs with JSON format metadata. Empty when no revision IDs are found.

    Raises:
        None. This helper is tolerant of missing fields and returns an empty list
        when the expected hierarchy is absent.

    Side Effects:
        None. Pure transformation of the provided response payload.
    """
    revision_ids: set[int] = set()

    def _walk_revisions(node: object) -> None:
        """
        Walk nested revision nodes to collect all integer revision IDs.
        """
        if isinstance(node, Mapping):
            typed_node = cast(Mapping[str, object], node)
            revision_value = typed_node.get("revisionID")
            if isinstance(revision_value, int):
                revision_ids.add(revision_value)

            children = typed_node.get("children")
            if isinstance(children, list):
                # Descend into child nodes that may contain deeper revisions.
                child_nodes = cast(list[object], children)
                for child in child_nodes:
                    _walk_revisions(child)

            nested_revisions = typed_node.get("revisions")
            if isinstance(nested_revisions, list):
                # Inspect nested revision collections for additional section IDs.
                nested_revision_nodes = cast(list[object], nested_revisions)
                for child in nested_revision_nodes:
                    _walk_revisions(child)

        elif isinstance(node, list):
            # Iterate list containers emitted by the API to reach nested dicts.
            list_nodes = cast(list[object], node)
            for child in list_nodes:
                _walk_revisions(child)

    response_payload = perma_response.get("response")
    if not isinstance(response_payload, Mapping):
        return []

    # Explore each artifact payload under the response to gather revision IDs.
    typed_response = cast(Mapping[str, object], response_payload)
    for artifact_payload in typed_response.values():
        _walk_revisions(artifact_payload)

    # Emit deterministic candidates so downstream ordering remains stable.
    return [
        DownloadCandidate(
            format="application/json",
            url=f"https://www.ck12.org/flx/get/detail/revision/{revision_id}?tiny=true",
            size=None,
        )
        for revision_id in sorted(revision_ids)
    ]


def collect_revision_candidates_with_skip_reason(
    identifier: str, perma_response: Mapping[str, object]
) -> tuple[list[DownloadCandidate], tuple[str, str] | None]:
    """
    Collect revision download candidates while surfacing a skip reason when none exist.

    Purpose:
        Wrap revision candidate extraction so callers (including the CLI) can emit a
        deterministic skip reason when the Perma payload lacks revisions or children.

    Args:
        identifier (str): Stable CK-12 slug used for logging and skip reporting.
        perma_response (Mapping[str, object]): Parsed Perma API payload to inspect.

    Returns:
        tuple[list[DownloadCandidate], tuple[str, str] | None]: Candidate list plus an
            optional skip reason tuple of (identifier, reason) when no candidates are
            discoverable.

    Raises:
        None. This helper is tolerant of missing fields and reports skip context
        instead of failing.

    Side Effects:
        None. Pure function returning extraction results and skip metadata.
    """
    candidates = extract_revision_download_candidates(perma_response)

    # Surface a structured skip reason so downstream logging can inform operators.
    if not candidates:
        return [], (identifier, "no revisions in perma response")

    return candidates, None


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

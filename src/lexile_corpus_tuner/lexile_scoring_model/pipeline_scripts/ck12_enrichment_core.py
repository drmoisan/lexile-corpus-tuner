"""Core CK-12 enrichment helpers shared by CLI and tests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .oer_models import DownloadCandidate

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
    language_hint = normalize_language_code(
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
                metadata["grade"] = normalize_text(
                    _extract_field(typed_node, ["educationalLevel", "grade"])
                )

            if metadata["language"] is None:
                metadata["language"] = normalize_language_code(
                    _extract_field(typed_node, ["inLanguage", "language"])
                )

    # Scan meta tags for remaining gaps such as author or locale hints.
    for meta_tag in soup.find_all("meta"):
        raw_name = meta_tag.get("name") or meta_tag.get("property")
        name_attr = (_get_first_str(raw_name) or "").lower()

        content = normalize_text(_get_first_str(meta_tag.get("content")))
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

        # Update language metadata if a locale tag is present.
        if metadata["language"] is None and name_attr in {
            "language",
            "content-language",
            "dc.language",
        }:
            metadata["language"] = normalize_language_code(content)

    # Fallback heuristic: read visible text for "Grade" labels.
    if metadata["grade"] is None:
        grade_pattern = re.compile(r"Grade\\s*([\\d-]+)", re.IGNORECASE)
        text_content = soup.get_text(" ", strip=True)
        grade_match = grade_pattern.search(text_content)
        if grade_match:
            metadata["grade"] = normalize_text(grade_match.group(1))

    return metadata


def extract_pdf_url(html: str) -> str | None:
    """
    Extract a PDF download URL from FlexBook HTML.

    Purpose:
        Identify the most likely PDF URL from anchor tags in the FlexBook HTML
        and return it for enrichment when available.

    Args:
        html (str): HTML document retrieved from the FlexBook page.

    Returns:
        str | None: Best candidate PDF URL, or None when no PDF link is found.

    Raises:
        ValueError: If a candidate URL is not absolute or does not point to a PDF.

    Side Effects:
        None. Pure extraction helper.
    """
    soup = BeautifulSoup(html, "html.parser")
    pdf_candidates: list[str] = []
    seen: set[str] = set()

    # Scan anchor/button elements for href-like attributes pointing to PDF exports.
    for tag in soup.find_all(["a", "button"]):
        for attr in ("href", "data-href", "data-url"):
            raw_value = _get_first_str(tag.get(attr))
            candidate = normalize_text(raw_value)
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

    """
    revision_ids: set[int] = set()

    def _walk_revisions(node: object) -> None:
        """
        Walk nested revision nodes to collect all integer revision IDs.

        The CK-12 Perma API returns revision IDs in multiple forms:
        - As `revisionID` or `artifactRevisionID` integer fields in mapping nodes
        - As plain integers directly in `children` lists (leaf section IDs)
        """
        if isinstance(node, Mapping):
            typed_node = cast(Mapping[str, object], node)

            # Collect revisionID when present (used by some artifact payloads).
            revision_value = typed_node.get("revisionID")
            if isinstance(revision_value, int):
                revision_ids.add(revision_value)

            # Collect artifactRevisionID (primary form in Perma responses).
            artifact_revision_value = typed_node.get("artifactRevisionID")
            if isinstance(artifact_revision_value, int):
                revision_ids.add(artifact_revision_value)

            children = typed_node.get("children")
            if isinstance(children, list):
                # Children may be nested objects or plain integer revision IDs.
                child_nodes = cast(list[object], children)
                for child in child_nodes:
                    if isinstance(child, int):
                        # Leaf section: child is a plain revision ID integer.
                        revision_ids.add(child)
                    else:
                        # Nested object: descend to extract further IDs.
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

    """
    candidates = extract_revision_download_candidates(perma_response)

    # Surface a structured skip reason so downstream logging can inform operators.
    if not candidates:
        return [], (identifier, "no revisions in perma response")

    return candidates, None


def _get_first_str(value: str | list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def normalize_language_code(value: str | None) -> str | None:
    cleaned = normalize_text(value)
    if cleaned is None:
        return None
    lower = cleaned.lower()
    if lower.startswith("en"):
        return "en"
    if lower.startswith("es"):
        return "es"
    return cleaned


def _extract_author_from_ldjson(node: dict[str, object]) -> str | None:
    author_field = node.get("author")
    if isinstance(author_field, str):
        return normalize_text(author_field)
    if isinstance(author_field, dict):
        author_data = cast(dict[str, object], author_field)
        return normalize_text(cast("str | None", author_data.get("name")))
    if isinstance(author_field, list):
        for author_candidate in cast(list[object], author_field):
            if isinstance(author_candidate, str):
                return normalize_text(author_candidate)
            if isinstance(author_candidate, dict):
                author_data = cast(dict[str, object], author_candidate)
                name_value = normalize_text(cast("str | None", author_data.get("name")))
                if name_value:
                    return name_value
    return None


def _extract_field(node: dict[str, object], keys: list[str]) -> str | None:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str):
            normalized_value = normalize_text(value)
            if normalized_value:
                return normalized_value
        if isinstance(value, list):
            for item in cast(list[object], value):
                if isinstance(item, str):
                    normalized_item = normalize_text(item)
                    if normalized_item:
                        return normalized_item
    return None

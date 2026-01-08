"""
CK-12 FlexBook catalog scaffold with typed interfaces for scraping.

Purpose:
    Provide a strongly-typed surface for fetching, parsing, and writing CK-12
    catalog data so downstream tasks can be implemented incrementally without
    breaking contracts.

Usage:
    Future implementations will call `fetch_catalog_page` to retrieve HTML,
    pass the result to `parse_catalog_rows`, and persist entries via
    `write_catalog_jsonl`. The `build_ck12_catalog` CLI will orchestrate those
    steps for pipeline operators.

Flow:
    1) Fetch CK-12 catalog HTML from the FlexBook browse endpoint.
    2) Parse book rows into `CatalogEntry` models with stable identifiers.
    3) Write deterministic JSONL output for later enrichment and curation.

Invariants / Constraints:
    - Slugs must remain stable for deduplication.
    - Output ordering should be deterministic once parsing is implemented.
    - Network requests and filesystem writes must be wrapped for testability.

Side Effects:
    Implementation will perform HTTP requests and filesystem writes; this stub
    performs no I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import typer

from .oer_models import CatalogEntry, generate_stable_slug

# CK-12 catalog is a static JSON file hosted on S3/CloudFront
# Discovered via browser DevTools inspection of bundle.js (Issue #73, January 2026)
# The endpoint was found by downloading the site's bundle.js, extracting API patterns,
# and identifying this stable JSON catalog endpoint containing all FlexBook metadata.
DEFAULT_CK12_CATALOG_URL = "https://static.ck12.org/testimonial/fbbrowse-prod.json"
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

app = typer.Typer(
    help="Build CK-12 catalog entries from the CK-12 FlexBook browse page."
)


def fetch_catalog_page(url: str) -> dict[str, Any]:
    """
    Retrieve the CK-12 catalog JSON for subsequent parsing.

    Purpose:
        Fetch the static JSON file containing the complete CK-12 FlexBook catalog.
        The JSON endpoint was discovered via browser DevTools inspection (Issue #73).
        Includes browser-like headers to avoid 403 Forbidden responses.

    Args:
        url (str): Fully-qualified catalog JSON URL to fetch.

    Returns:
        dict[str, Any]: Parsed JSON catalog data from CK-12.

    Raises:
        RuntimeError: If request fails, returns non-success, or JSON parse fails.
        requests.RequestException: For lower-level network errors.

    Side Effects:
        Issues an HTTP GET request with timeouts and browser headers.
    """
    try:
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timed out while fetching CK-12 catalog from {url}"
        ) from exc
    except requests.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON received from CK-12 catalog at {url}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch CK-12 catalog from {url}") from exc


def parse_catalog_json(catalog_data: dict[str, Any]) -> list[CatalogEntry]:
    """
    Convert CK-12 catalog JSON into typed catalog entries.

    Purpose:
        Transform JSON catalog data into `CatalogEntry` objects with stable identifiers
        derived from FlexBook URLs, ensuring downstream enrichment and curation
        operate on consistent data.

    Args:
        catalog_data (dict[str, Any]): JSON catalog data from CK-12 endpoint.
            Expected structure: {"books": [{"Title": "...", "Content_URL": "..."}]}

    Returns:
        list[CatalogEntry]: Parsed catalog entries in JSON order.

    Raises:
        ValueError: If required fields are missing or malformed.

    Side Effects:
        None. Pure transformation of provided JSON.
    """
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()

    # The JSON has a "books" key containing the list of FlexBooks
    books_raw = catalog_data.get("books", [])

    if not isinstance(books_raw, list):
        raise ValueError(
            f"Expected catalog data to contain 'books' list, got {type(books_raw)}"
        )

    # Type cast to help Pyright understand this is a list of dicts
    books: list[dict[str, object]] = books_raw  # type: ignore[assignment]

    for book in books:
        # Extract book URL from Content_URL field
        # CK-12 uses both /cbook/ (FlexBooks) and /book/ (older format) patterns
        book_url_raw = book.get("Content_URL", "")
        if not isinstance(book_url_raw, str):
            continue
        book_url: str = book_url_raw

        if not book_url or ("/cbook/" not in book_url and "/book/" not in book_url):
            continue  # Skip entries without valid book URLs

        # Generate stable identifier from URL slug
        slug = _extract_slug_from_url(book_url)
        identifier = generate_stable_slug(slug)
        if identifier in seen_ids:
            continue

        # Extract title and metadata from JSON fields
        title_raw = book.get("Title", slug)
        title: str = title_raw if isinstance(title_raw, str) else slug

        language_code_raw = book.get("Language_Code", "")
        language_code: str = (
            language_code_raw if isinstance(language_code_raw, str) else ""
        )
        language_list: list[str] = [language_code] if language_code else []

        entries.append(
            CatalogEntry(
                source_id="ck12",
                identifier=identifier,
                title=title,
                creator=None,  # CK-12 JSON doesn't include author field
                year=None,  # CK-12 JSON doesn't include publication year
                language=language_list,
                license_url=None,  # CK-12 license info not in catalog JSON
            )
        )
        seen_ids.add(identifier)

    return entries


def _extract_slug_from_url(url: str) -> str:
    """
    Pull the terminal path segment from a CK-12 FlexBook URL.

    Purpose:
        Normalize FlexBook URLs into stable slug seeds before passing through
        `generate_stable_slug`, guaranteeing deterministic identifiers.

    Args:
        url (str): Absolute or relative FlexBook URL.

    Returns:
        str: The last non-empty path segment to be slugified.

    Raises:
        ValueError: When no valid path segment is present.
    """
    parsed = urlparse(url)
    # Extract the last non-empty segment from the path.
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        raise ValueError(f"CK-12 URL missing path segment: {url}")
    slug = segments[-1]
    if not slug:
        raise ValueError(f"CK-12 URL missing slug: {url}")
    return slug


def write_catalog_jsonl(rows: list[CatalogEntry], path: Path) -> None:
    """
    Persist catalog entries to a JSONL file.

    Purpose:
        Provide a deterministic writer that can be reused by the CLI and tests,
        ensuring stable ordering and atomic replacement.

    Args:
        rows (list[CatalogEntry]): Catalog entries ready for serialization.
        path (Path): Target file path for JSONL output.

    Returns:
        None

    Raises:
        OSError: If writing to the target path fails.

    Side Effects:
        Creates parent directories as needed and replaces any existing file via
        a temporary write + rename to avoid partial output.
    """
    ordered_rows = sorted(rows, key=lambda entry: (entry.identifier, entry.title or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        # Write entries in identifier order to keep JSONL output deterministic.
        for entry in ordered_rows:
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
                            candidate.__dict__
                            for candidate in entry.download_candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temp_path.replace(path)


@app.command()
def build_ck12_catalog(
    catalog_url: str = typer.Option(  # noqa: B008 - Typer framework pattern
        DEFAULT_CK12_CATALOG_URL,
        help="CK-12 FlexBook catalog URL to scrape.",
    ),
    out_dir: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/meta/catalogs"),
        help="Directory where ck12_catalog.jsonl will be written.",
    ),
) -> None:
    """
    CLI entry point for building the CK-12 catalog JSONL artifact.

    Purpose:
        Orchestrate fetch, parse, and write steps so pipeline operators can
        produce a refreshed CK-12 catalog with a single command.

    Args:
        catalog_url (str): CK-12 catalog JSON URL to fetch.
        out_dir (Path): Output directory for the catalog JSONL file.

    Returns:
        None

    Side Effects:
        Fetches remote JSON catalog and writes to the filesystem.
    """
    catalog_json = fetch_catalog_page(catalog_url)
    entries = parse_catalog_json(catalog_json)
    output_path = out_dir / "ck12_catalog.jsonl"
    write_catalog_jsonl(entries, output_path)
    typer.echo(f"Wrote {len(entries)} CK-12 entries to {output_path}")


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

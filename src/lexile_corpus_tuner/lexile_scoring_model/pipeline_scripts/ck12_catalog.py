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
from urllib.parse import urljoin, urlparse

import requests
import typer
from bs4 import BeautifulSoup

from .oer_models import CatalogEntry, generate_stable_slug

DEFAULT_CK12_CATALOG_URL = "https://www.ck12.org/fbbrowse/list?grade=all%20grades&language=all%20languages&subject=all%20subjects"
REQUEST_TIMEOUT_SECONDS = 30

app = typer.Typer(
    help="Build CK-12 catalog entries from the CK-12 FlexBook browse page."
)


def fetch_catalog_page(url: str) -> str:
    """
    Retrieve the CK-12 catalog HTML for subsequent parsing.

    Purpose:
        Provide a dedicated hook for HTTP retrieval so the implementation can
        centralize error handling and allow injection/mocking in tests.

    Args:
        url (str): Fully-qualified catalog URL to fetch.

    Returns:
        str: Raw HTML content of the CK-12 catalog page.

    Raises:
        RuntimeError: If the request fails or returns a non-success status code.
        requests.RequestException: For lower-level network errors.

    Side Effects:
        Implementation will issue an HTTP GET request with timeouts.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timed out while fetching CK-12 catalog from {url}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch CK-12 catalog from {url}") from exc

    return response.text


def parse_catalog_rows(html: str) -> list[CatalogEntry]:
    """
    Convert CK-12 catalog HTML into typed catalog entries.

    Purpose:
        Transform raw HTML into `CatalogEntry` objects with stable identifiers
        derived from FlexBook URLs, ensuring downstream enrichment and curation
        operate on consistent data.

    Args:
        html (str): HTML document retrieved from the CK-12 catalog.

    Returns:
        list[CatalogEntry]: Parsed catalog entries preserving page ordering.

    Raises:
        ValueError: If a FlexBook link is malformed or lacks a slug.

    Side Effects:
        None. Pure transformation of provided HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()

    # Iterate FlexBook links that point to cbook pages, including relative URLs.
    for link in soup.select("a[href]"):
        href_value = link.get("href", "")
        if not isinstance(href_value, str):
            continue

        # Only accept FlexBook links; the page also includes navigation and asset links.
        if "/cbook/" not in href_value:
            continue

        resolved_url = urljoin(DEFAULT_CK12_CATALOG_URL, href_value)
        slug = _extract_slug_from_url(resolved_url)
        identifier = generate_stable_slug(slug)
        # Deduplicate repeated links to keep output stable and unique.
        if identifier in seen_ids:
            continue

        # Preserve a human-readable title; fall back to slug if the link text is empty.
        title_text = link.get_text(strip=True) or slug
        entries.append(
            CatalogEntry(
                source_id="ck12",
                identifier=identifier,
                title=title_text,
                creator=None,
                year=None,
                language=[],
                license_url=None,
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
        catalog_url (str): CK-12 catalog URL to fetch.
        out_dir (Path): Output directory for the catalog JSONL file.

    Returns:
        None

    Side Effects:
        Implementation will fetch remote HTML and write to the filesystem.
    """
    catalog_html = fetch_catalog_page(catalog_url)
    entries = parse_catalog_rows(catalog_html)
    output_path = out_dir / "ck12_catalog.jsonl"
    write_catalog_jsonl(entries, output_path)
    typer.echo(f"Wrote {len(entries)} CK-12 entries to {output_path}")


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

"""
Catalog builder for OpenStax and CK-12 titles via Internet Archive search.

Purpose:
    Build catalog JSONL files containing base metadata and placeholders for
    download candidates. This is the first stage in the OER pipeline.

Usage:
    The CLI command `build_oer_catalog` queries IA for each requested source,
    parses results into CatalogEntry objects, and writes JSONL files under
    the target directory.

Flow:
    1) Construct a stable IA query per source.
    2) Fetch search results (scrape API recommended, but this module keeps the
       interface simple for mocking/testing).
    3) Parse each row into CatalogEntry objects with empty download_candidates
       (to be enriched later).
    4) Write JSONL to disk.

Constraints:
    - Supported sources: openstax, ck12.
    - Network calls should be wrapped/mocked in tests to avoid external I/O.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

import typer

from .oer_models import CatalogEntry

app = typer.Typer(help="Build OER catalog entries from Internet Archive search.")

IA_SCRAPE_ENDPOINT = "https://archive.org/services/search/v1/scrape"
SUPPORTED_SOURCES = {"openstax", "ck12"}
DEFAULT_FIELDS = "identifier,title,creator,year,language,licenseurl,publicdate"


def build_ia_query(source: str) -> str:
    """
    Build an IA query string tuned for the requested source.

    Purpose:
        Encapsulate query logic so tests can verify coverage of source-specific
        filters without hitting the network.

    Args:
        source: Source key, expected "openstax" or "ck12".

    Returns:
        An IA query string suitable for the scrape API.

    Raises:
        ValueError: When an unsupported source is provided.
    """
    normalized = source.strip().lower()
    if normalized not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}")
    if normalized == "openstax":
        return (
            '(mediatype:texts) AND (openstax OR "OpenStax") AND '
            '(textbook OR "open textbook") AND NOT (ck12 OR "CK-12")'
        )
    return (
        '(mediatype:texts) AND ("CK-12" OR ck12 OR "CK12" OR "ck-12") AND '
        '(textbook OR "flexbook" OR "FlexBook") AND NOT (openstax OR "OpenStax")'
    )


def _http_get_json(
    url: str,
) -> dict[
    str, object
]:  # pragma: no cover - network helper exercised via higher-level mocks
    """Perform a GET and return parsed JSON.

    This helper is isolated to simplify mocking in tests and to centralize
    error handling.
    """
    req = urllib.request.Request(url)  # noqa: S310 - expected IA HTTPS request
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        payload = resp.read()
    return cast(dict[str, object], json.loads(payload.decode("utf-8", errors="ignore")))


def fetch_ia_search_results(
    query: str,
    *,
    fields: str = DEFAULT_FIELDS,
    count: int = 200,
    fetcher: Callable[[str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """
    Fetch IA search results using the scrape endpoint.

    Args:
        query: IA query string.
        fields: Comma-separated field list.
        count: Page size for scrape API.
        fetcher: Optional override for HTTP GET returning JSON (for testing).

    Returns:
        List of raw result dicts returned by the IA service.
    """
    http: Callable[[str], dict[str, object]] = fetcher or _http_get_json
    cursor: str | None = None
    items: list[dict[str, object]] = []
    # Iterate through cursors until exhaustion to collect all items deterministically.
    # Walk through cursor pages until IA indicates completion.
    while True:
        params = {
            "q": query,
            "fields": fields,
            "count": str(count),
        }
        if cursor:
            params["cursor"] = cursor
        url = IA_SCRAPE_ENDPOINT + "?" + urllib.parse.urlencode(params)
        page = http(url)
        page_items = cast(list[dict[str, object]], page.get("items") or [])
        # Extend the running list while preserving ordering for stable output.
        items.extend(page_items)
        cursor = cast(str | None, page.get("cursor"))
        if not cursor:
            break
    return items


def parse_catalog_entry(raw: Mapping[str, object], source_id: str) -> CatalogEntry:
    """
    Convert an IA result row into a CatalogEntry.

    Missing optional fields are represented as None; download_candidates are
    initialized empty and populated during enrichment.
    """
    title_raw = raw.get("title")
    title = str(title_raw) if title_raw is not None else None
    creator = _first_or_none(raw.get("creator"))
    year_raw = raw.get("year")
    year = str(year_raw) if year_raw is not None else None
    language_raw = raw.get("language")
    language = _ensure_list_str(language_raw)
    license_raw = raw.get("licenseurl")
    license_url = str(license_raw) if license_raw is not None else None
    identifier = raw.get("identifier")
    if not isinstance(identifier, str):
        raise ValueError("Result missing identifier")
    return CatalogEntry(
        source_id=source_id,
        identifier=identifier,
        title=title,
        creator=creator,
        year=year,
        language=language,
        license_url=license_url,
        download_candidates=[],
    )


def write_catalog_jsonl(entries: list[CatalogEntry], output_path: Path) -> None:
    """Write CatalogEntry items to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
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
def build_oer_catalog(  # pragma: no cover - CLI wrapper
    # noqa: B008 - Typer option defaults are runtime configuration
    sources: str = typer.Option(
        "openstax,ck12", help="Comma-separated list of sources to query"
    ),  # noqa: B008 - Typer options evaluated at import for CLI metadata
    out_dir: Path = typer.Option(  # noqa: B008 - Typer options evaluated at import for CLI metadata
        Path("data/meta/catalogs"), "--out-dir", help="Directory for JSONL output"
    ),
) -> None:
    """
    CLI entrypoint to build catalog JSONL files for requested sources.

    Post-filters results to ensure source exclusivity since IA query NOT operators
    are unreliable and metadata tagging is inconsistent.
    Processes OpenStax first to build a reference set of known identifiers.
    """
    selected_sources = [s.strip().lower() for s in sources.split(",") if s.strip()]

    # Track OpenStax identifiers for exclusion from other sources
    openstax_identifiers: set[str] = set()

    for src in selected_sources:
        if src not in SUPPORTED_SOURCES:
            typer.echo(f"Skipping unsupported source: {src}", err=True)
            continue
        query = build_ia_query(src)
        typer.echo(f"Querying IA for {src} with: {query}")
        results = fetch_ia_search_results(query)
        typer.echo(f"Found {len(results)} raw results for {src}")
        entries: list[CatalogEntry] = []
        filtered_count = 0
        # Parse each result and post-filter to ensure source exclusivity.
        # IA NOT operators and metadata tagging are unreliable,
        # so apply exclusion logic here.
        for raw in results:
            identifier = str(raw.get("identifier", ""))
            title_lower = str(raw.get("title", "")).lower()
            creator_lower = str(raw.get("creator", "")).lower()

            # Post-filter: check for conflicting source markers
            if src == "ck12":
                # Reject any result with OpenStax markers or OpenStax identifiers
                if (
                    identifier in openstax_identifiers
                    or "openstax" in title_lower
                    or "openstax" in creator_lower
                    or "openstax" in identifier.lower()
                ):
                    filtered_count += 1
                    continue
            elif src == "openstax":
                # Reject any result with CK-12 markers
                if (
                    "ck12" in title_lower
                    or "ck-12" in title_lower
                    or "ck 12" in title_lower
                    or "flexbook" in title_lower
                    or "ck12" in identifier.lower()
                    or "ck-12" in identifier.lower()
                ):
                    filtered_count += 1
                    continue
            try:
                entries.append(parse_catalog_entry(raw, src))
            except ValueError as exc:
                typer.echo(f"Skipping row without identifier: {exc}", err=True)

        # After processing OpenStax, save its identifiers for filtering other sources
        if src == "openstax":
            openstax_identifiers = {entry.identifier for entry in entries}
            typer.echo(
                f"Collected {len(openstax_identifiers)} OpenStax identifiers "
                f"for exclusion filtering"
            )

        if filtered_count > 0:
            typer.echo(f"Filtered out {filtered_count} conflicting entries for {src}")
        output_path = out_dir / f"{src}_catalog.jsonl"
        write_catalog_jsonl(entries, output_path)
        typer.echo(f"Wrote {len(entries)} entries to {output_path}")


def _first_or_none(value: Any) -> str | None:
    """Return the first string from a list-or-string input."""
    if value is None:
        return None
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return str(list_value[0]) if list_value else None
    return str(value)


def _ensure_list_str(value: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [str(item) for item in list_value]
    return [str(value)]


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

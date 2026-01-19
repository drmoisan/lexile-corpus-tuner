"""
CK-12 FlexBook catalog scaffold with typed interfaces for scraping.

Purpose:
    Provide a strongly-typed surface for fetching, parsing, and writing CK-12
    catalog data so downstream tasks can be implemented incrementally without
    breaking contracts.

Usage:
    Call `fetch_catalog_page` to retrieve Browse API JSON, pass the result to
    `parse_catalog_json`, and persist entries via `write_catalog_jsonl`. The
    `build_ck12_catalog` CLI orchestrates those steps for pipeline operators.

Flow:
    1) Fetch CK-12 catalog JSON from the FlexBook browse endpoint.
    2) Parse book rows into `CatalogEntry` models with stable identifiers.
    3) Write deterministic JSONL output for later enrichment and curation.

Invariants / Constraints:
    - Slugs must remain stable for deduplication.
    - Output ordering should be deterministic once parsing is implemented.
    - Network requests and filesystem writes must be wrapped for testability.

Side Effects:
    Performs HTTP requests and filesystem writes when invoked via CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import typer

from .oer_models import CatalogEntry, generate_stable_slug

# CK-12 static FlexBook browse feed (Issue #73 spec)
DEFAULT_CK12_CATALOG_URL = "https://static.ck12.org/testimonial/fbbrowse-prod.json"
REQUEST_TIMEOUT_SECONDS = 30

# HTTP headers to mimic a real browser and avoid 403 Forbidden (Issue #73)
REQUEST_HEADERS = {
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
    help="Build CK-12 catalog entries from the CK-12 FlexBook browse page."
)


def extract_slug_from_content_url(url: str) -> str | None:
    """
    Derive the CK-12 content slug from known Content_URL patterns.

    Purpose:
        Extract a stable slug from CK-12 Content_URL values so identifier derivation
        remains deterministic across static feed and Browse API inputs.

    Args:
        url (str): Fully-qualified CK-12 Content_URL string to parse.

    Returns:
        str | None: Slug when the URL matches /cbook/, /user:<handle>/cbook/, or
            /book/ patterns; None when the URL cannot be parsed.

    Raises:
        None.

    Side Effects:
        None.
    """
    parsed = urlparse(url)
    # Normalize path segments to enable consistent positional matching.
    path_parts = [segment for segment in parsed.path.split("/") if segment]

    if parsed.netloc == "flexbooks.ck12.org":
        if "cbook" in path_parts:
            cbook_index = path_parts.index("cbook")
            if cbook_index + 1 < len(path_parts):
                return path_parts[cbook_index + 1]
        return None

    if parsed.netloc == "www.ck12.org":
        if "book" in path_parts:
            book_index = path_parts.index("book")
            if book_index + 1 < len(path_parts):
                return path_parts[book_index + 1]
        return None

    return None


def fetch_catalog_page(url: str) -> dict[str, Any]:
    """
    Retrieve the CK-12 catalog JSON for subsequent parsing.

    Purpose:
         Fetch the Browse API JSON containing the CK-12 FlexBook catalog using the
         required browser-like headers to avoid 403 Forbidden responses.

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
        json_data = response.json()
        if not isinstance(json_data, dict):
            raise RuntimeError(
                f"Unexpected CK-12 catalog payload type {type(json_data)} from {url}"
            )
        return cast(dict[str, Any], json_data)
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timed out while fetching CK-12 catalog from {url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON received from CK-12 catalog at {url}"
        ) from exc
    except ValueError as exc:
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
        catalog_data (dict[str, Any]): JSON catalog data from CK-12 Browse API.
            Expected structures include:
                - {"books": [{...}]}
                - {"response": {"items": [{...}]}}

    Returns:
        list[CatalogEntry]: Parsed catalog entries in JSON order.

    Raises:
        ValueError: If required fields are missing or malformed.

    Side Effects:
        None. Pure transformation of provided JSON.
    """
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()

    # Locate the list of FlexBooks in either the legacy "books" key or the
    # Browse API "response.flexbook" / "response.items" shape.
    books_raw_obj: list[dict[str, object]] | dict[str, object] | None
    books_from_root: object | None = catalog_data.get("books")
    if isinstance(books_from_root, list):
        root_books_list = cast(list[object], books_from_root)
        validated_list: list[dict[str, object]] = []
        for item in root_books_list:
            if not isinstance(item, dict):
                raise ValueError(
                    "Expected each catalog entry to be a dict when parsing CK-12 "
                    "catalog"
                )
            validated_list.append(cast(dict[str, object], item))
        books_raw_obj = validated_list
    elif isinstance(books_from_root, dict):
        books_raw_obj = cast(dict[str, object], books_from_root)
    elif books_from_root is None:
        response_obj = catalog_data.get("response")
        if isinstance(response_obj, dict):
            # Treat nested JSON objects as dict[str, Any] so `dict.get()` is typed and
            # downstream parsing can narrow values safely.
            response_dict = cast(dict[str, Any], response_obj)
            # Browse API returns data under "flexbook" key (observed Jan 2026).
            nested_books: object | None = (
                response_dict.get("flexbook")
                or response_dict.get("items")
                or response_dict.get("books")
            )
            if isinstance(nested_books, list):
                nested_books_list = cast(list[object], nested_books)
                validated_list: list[dict[str, object]] = []
                for item in nested_books_list:
                    if not isinstance(item, dict):
                        raise ValueError(
                            "Expected each catalog entry to be a dict when parsing"
                            " CK-12 catalog"
                        )
                    validated_list.append(cast(dict[str, object], item))
                books_raw_obj = validated_list
            elif isinstance(nested_books, dict):
                books_raw_obj = cast(dict[str, object], nested_books)
            else:
                books_raw_obj = None
        else:
            books_raw_obj = None
    else:
        raise ValueError(
            "Expected catalog data to contain 'books' list or dict in the root payload"
        )
    if isinstance(books_raw_obj, dict) and "items" in books_raw_obj:
        # Preserve Browse API responses where the payload nests items inside
        # response.flexbook.items rather than a top-level list.
        response_flexbook_dict = books_raw_obj
        candidate_items = response_flexbook_dict.get("items")
        if isinstance(candidate_items, list):
            nested_items_list = cast(list[object], candidate_items)
            validated_items: list[dict[str, object]] = []
            for item in nested_items_list:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Expected each catalog entry to be a dict when parsing CK-12 "
                        "catalog"
                    )
                validated_items.append(cast(dict[str, object], item))
            books_raw_obj = validated_items
    books_raw_list: list[dict[str, object]]
    if books_raw_obj is None:
        books_raw_list = []
    elif isinstance(books_raw_obj, list):
        books_raw_list = books_raw_obj
    else:
        raise ValueError(
            "Expected catalog data to contain 'books' list, got "
            f"{type(books_raw_obj).__name__}"
        )

    books: list[dict[str, object]] = books_raw_list

    # Convert each Browse API item into a CatalogEntry using canonical handles.
    # Filter out rows that are missing required identifiers to keep output
    # deterministic.
    for book in books:
        is_static_feed_entry = "Language" in book
        artifact_id: object | None = book.get("artifactID") or book.get("artifactId")
        artifact_type: object | None = book.get("artifactType") or book.get(
            "artifact_type"
        )
        content_url_raw: object | None = book.get("Content_URL")
        content_url_host: str | None = (
            urlparse(content_url_raw).netloc
            if isinstance(content_url_raw, str) and content_url_raw
            else None
        )
        content_url_slug: str | None = (
            extract_slug_from_content_url(content_url_raw)
            if isinstance(content_url_raw, str) and content_url_raw
            else None
        )
        handle_raw: object | None = book.get("handle")
        handle: str | None = None

        if is_static_feed_entry:
            # Static feed identifiers prefer Content_URL slug, then handle, then Title.
            identifier_source_raw: object | None = (
                content_url_slug or handle_raw or book.get("Title")
            )
            if not isinstance(identifier_source_raw, str) or not identifier_source_raw:
                continue
            identifier = generate_stable_slug(identifier_source_raw)
            if identifier in seen_ids:
                continue

            artifact_id_value: int | None = None
            if isinstance(artifact_id, int):
                artifact_id_value = artifact_id
            elif isinstance(artifact_id, str) and artifact_id.isdigit():
                artifact_id_value = int(artifact_id)

            if content_url_host == "flexbooks.ck12.org":
                artifact_type_value = "flexbook"
            elif content_url_host == "www.ck12.org":
                artifact_type_value = "book"
            else:
                artifact_type_value = artifact_type
        else:
            if artifact_id is None or not isinstance(artifact_id, int | str):
                continue
            if not isinstance(artifact_type, str) or not artifact_type:
                continue
            if not isinstance(handle_raw, str) or not handle_raw:
                continue
            handle = handle_raw

            identifier = generate_stable_slug(handle)
            if identifier in seen_ids:
                continue

            artifact_id_value: int | None = None
            if isinstance(artifact_id, int):
                artifact_id_value = artifact_id
            else:
                # Preserve the previous behavior (numeric strings become ints) while
                # tolerating non-string truthy values by stringifying them first.
                artifact_id_text = str(artifact_id)
                if artifact_id_text.isdigit():
                    artifact_id_value = int(artifact_id_text)

            artifact_type_text = str(artifact_type)
            if content_url_host == "flexbooks.ck12.org":
                artifact_type_value = "flexbook"
            elif content_url_host == "www.ck12.org":
                artifact_type_value = "book"
            else:
                artifact_type_value = artifact_type_text if artifact_type_text else None

        # Extract title and metadata from JSON fields, defaulting to the identifier
        # so missing titles remain deterministic.
        title_raw = (
            book.get("Title") or book.get("title") or book.get("name") or identifier
        )
        title: str = title_raw if isinstance(title_raw, str) else identifier

        language_list: list[str] = []
        language_field = (
            book.get("Language")
            if is_static_feed_entry
            else book.get("Language_Code") or book.get("language")
        )
        if isinstance(language_field, str) and language_field:
            language_list = [language_field]
        elif isinstance(language_field, list):
            # Preserve language codes provided as a list without altering order.
            raw_lang_list = cast(list[object], language_field)
            language_list = [
                lang for lang in raw_lang_list if isinstance(lang, str) and lang
            ]

        entry_handle: str | None
        if not is_static_feed_entry:
            entry_handle = handle
        elif isinstance(handle_raw, str):
            entry_handle = handle_raw
        else:
            entry_handle = None

        if isinstance(artifact_type_value, str) or artifact_type_value is None:
            safe_artifact_type: str | None = artifact_type_value
        else:
            safe_artifact_type = None

        entries.append(
            CatalogEntry(
                source_id="ck12",
                identifier=identifier,
                title=title,
                creator=None,  # CK-12 JSON doesn't include author field
                year=None,  # CK-12 JSON doesn't include publication year
                language=language_list,
                license_url=None,  # CK-12 license info not in catalog JSON
                artifact_type=safe_artifact_type,
                handle=entry_handle,
                artifact_id=artifact_id_value,
            )
        )
        seen_ids.add(identifier)

    return entries


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
    # Keep output deterministic by sorting identifiers and titles before writing.
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
                        "artifact_type": entry.artifact_type,
                        "handle": entry.handle,
                        "artifact_id": entry.artifact_id,
                        # Serialize download candidates to dictionaries for JSON output.
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

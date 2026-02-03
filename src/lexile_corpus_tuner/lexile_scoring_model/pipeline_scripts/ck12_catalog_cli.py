"""CLI entry points for CK-12 catalog generation."""

from __future__ import annotations

from pathlib import Path

import typer

from .ck12_catalog import (
    DEFAULT_CK12_CATALOG_URL,
    fetch_catalog_page,
    parse_catalog_json,
    write_catalog_jsonl,
)

app = typer.Typer(
    help="Build CK-12 catalog entries from the CK-12 FlexBook browse page."
)


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

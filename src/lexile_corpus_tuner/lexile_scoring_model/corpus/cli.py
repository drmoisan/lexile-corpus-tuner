from __future__ import annotations

from pathlib import Path

import click

from . import download, frequencies, normalize


@click.group(name="corpus")
def corpus_group() -> None:
    """Commands for managing the proxy Lexile corpus."""


@corpus_group.command("download")
@click.option(
    "--gutenberg-limit",
    type=int,
    default=None,
    help="Max Gutenberg books to download (for testing).",
)
@click.option(
    "--sources",
    type=str,
    default=None,
    help="Comma-separated sources to download (e.g., gutenberg,simple_wiki,oer).",
)
def corpus_download(gutenberg_limit: int | None, sources: str | None) -> None:
    """Download raw corpus sources."""
    allowed = _parse_sources(sources)
    download.ensure_dirs()
    if allowed is None or "gutenberg" in allowed:
        download.download_gutenberg_subset(limit=gutenberg_limit)
    if allowed is None or "simple_wiki" in allowed:
        download.download_simple_wiki_dump()
    if allowed is None or "oer" in allowed:
        download.download_oer_sources()


@corpus_group.command("normalize")
@click.option(
    "--shard-size-tokens",
    type=int,
    default=100_000,
    show_default=True,
    help="Number of tokens per normalized shard before rolling over.",
)
@click.option(
    "--sources",
    type=str,
    default=None,
    help="Comma-separated sources to normalize (e.g., gutenberg,simple_wiki,oer).",
)
def corpus_normalize(shard_size_tokens: int, sources: str | None) -> None:
    """Normalize and tokenize raw corpora into shards."""
    allowed = _parse_sources(sources)
    normalize.normalize_all_sources(
        shard_size_tokens=shard_size_tokens, allowed_sources=allowed
    )


@corpus_group.command("frequencies")
@click.option(
    "--weighted",
    is_flag=True,
    help="Compute weighted word frequencies using a weight matrix config.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Path to weight config (json or yaml) containing weights.source.era entries.",
)
def corpus_frequencies(weighted: bool, config_path: Path | None) -> None:
    """Compute global word frequencies from normalized shards."""
    frequencies.compute_global_frequencies(weighted=weighted, config_path=config_path)


def _parse_sources(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}

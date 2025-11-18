from __future__ import annotations

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
def corpus_download(gutenberg_limit: int | None) -> None:
    """Download raw corpus sources."""
    download.ensure_dirs()
    download.download_gutenberg_subset(limit=gutenberg_limit)
    download.download_simple_wiki_dump()


@corpus_group.command("normalize")
@click.option(
    "--shard-size-tokens",
    type=int,
    default=100_000,
    show_default=True,
    help="Number of tokens per normalized shard before rolling over.",
)
def corpus_normalize(shard_size_tokens: int) -> None:
    """Normalize and tokenize raw corpora into shards."""
    normalize.normalize_all_sources(shard_size_tokens=shard_size_tokens)


@corpus_group.command("frequencies")
def corpus_frequencies() -> None:
    """Compute global word frequencies from normalized shards."""
    frequencies.compute_global_frequencies()

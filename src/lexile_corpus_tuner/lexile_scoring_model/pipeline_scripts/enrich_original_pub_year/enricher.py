"""
Core enrichment workflow for original publication year backfilling.

Purpose:
    Load parquet data, run provider queries, and emit enriched rows with inferred
    publication years.

Usage:
    Called by CLI entrypoints or other orchestrators to enrich a dataframe or
    parquet file using the configured providers.

Side Effects:
    Performs network I/O through provider clients, reads/writes parquet files, and
    persists cache/checkpoint data on disk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from .enrichment_result import EnrichmentResult
from .file_cache import FileCache
from .file_checkpoint import FileCheckpoint
from .match_result import MatchResult
from .match_utils import is_missing_year, normalize_text, select_best_match
from .noop_fallback import NoopFallback
from .open_library_client import OpenLibraryClient, OpenLibrarySearchError
from .summary import Summary


class EnrichmentError(Exception):
    """
    Raised when enrichment fails for a specific row.

    Purpose:
        Provide clear context about which row failed enrichment and why.

    Attributes:
        row_index (int): Zero-based row index in the dataframe.
        title (str): The title being enriched.
        author (str): The author being enriched.
        cause (Exception): The underlying exception that caused the failure.
    """

    def __init__(
        self, row_index: int, title: str, author: str, cause: Exception
    ) -> None:
        """
        Initialize the exception with row context.

        Args:
            row_index (int): Zero-based index of the row that failed.
            title (str): Title from the failing row.
            author (str): Author from the failing row.
            cause (Exception): The underlying error that triggered this.

        Side Effects:
            None.
        """
        self.row_index = row_index
        self.title = title
        self.author = author
        self.cause = cause
        message = (
            f"Enrichment failed at row {row_index} "
            f"(title='{title}', author='{author}'): {cause}"
        )
        super().__init__(message)


if TYPE_CHECKING:
    from pandas import DataFrame

    from .cache_store import CacheStore
    from .checkpoint_store import CheckpointStore
    from .enrichment_config import EnrichmentConfig
    from .fallback_client import FallbackClient
    from .match_result import Confidence


def enrich_dataframe(
    df: DataFrame,
    *,
    config: EnrichmentConfig,
    client: OpenLibraryClient,
    cache: CacheStore | None = None,
    checkpoint: CheckpointStore | None = None,
    fallback: FallbackClient | None = None,
) -> EnrichmentResult:
    """
    Enrich a dataframe in memory using cache, checkpoints, fuzzy matching, and optional
    fallback clients.

    Purpose:
        Orchestrate publication-year lookups for each row while supporting caching
        and resumability.

    Args:
        df (pd.DataFrame): Input dataframe containing title and author columns.
        config (EnrichmentConfig): Runtime tuning parameters for enrichment.
        client (OpenLibraryClient): Primary search client for Open Library.
        cache (CacheStore | None): Optional cache implementation; default creates a
            `FileCache`.
        checkpoint (CheckpointStore | None): Optional checkpoint store; default uses
            `FileCheckpoint`.
        fallback (FallbackClient | None): Optional secondary catalog client; defaults
            to `NoopFallback`.

    Returns:
        EnrichmentResult: Enriched dataframe plus processing summary metrics.

    Side Effects:
        May perform network I/O, disk reads/writes for cache and checkpoints, and
        sleeps for rate limiting.
    """

    working = df.copy()
    summary = Summary()
    cache_store = cache or FileCache(config.cache_dir)
    checkpoint_store = checkpoint or FileCheckpoint(config.checkpoint_path)
    fallback_client = fallback or NoopFallback()

    start_index = checkpoint_store.load()
    results_year: list[int | None] = []
    results_conf: list[Confidence] = []
    results_source: list[str | None] = []

    for idx, row in enumerate(working.itertuples(index=False), start=0):
        if idx < start_index:
            existing_year = getattr(row, "original_pub_year", None)
            missing_year = is_missing_year(existing_year)
            results_year.append(None if missing_year else existing_year)
            results_conf.append(getattr(row, "pub_year_confidence", "none"))
            results_source.append(getattr(row, "original_pub_source", None))
            continue

        title = getattr(row, "title", "") or ""
        authors = getattr(row, "authors", "") or ""
        normalized_title = normalize_text(title)
        normalized_author = normalize_text(authors)

        cache_key = f"{normalized_title}__{normalized_author}"
        cached = cache_store.get(cache_key)
        result: MatchResult
        if cached is not None:
            result = cached
        else:
            try:
                candidates = client.search(title, authors)
            except OpenLibrarySearchError:
                summary.record_error()
                result = MatchResult(
                    year=None, confidence="none", source="openlibrary_error"
                )
            else:
                result = select_best_match(
                    candidates=candidates,
                    normalized_title=normalized_title,
                    normalized_author=normalized_author,
                    threshold=config.fuzzy_threshold,
                    disable_fuzzy=config.disable_fuzzy,
                )
                if result.confidence == "none":
                    try:
                        fallback_candidates = fallback_client.search(title, authors)
                    except Exception:
                        summary.record_error()
                        result = MatchResult(
                            year=None,
                            confidence="none",
                            source="fallback_error",
                        )
                    else:
                        result = select_best_match(
                            candidates=fallback_candidates,
                            normalized_title=normalized_title,
                            normalized_author=normalized_author,
                            threshold=config.fuzzy_threshold,
                            disable_fuzzy=config.disable_fuzzy,
                        )
                if result.source not in {"openlibrary_error", "fallback_error"}:
                    cache_store.set(cache_key, result)

        results_year.append(result.year)
        results_conf.append(result.confidence)
        results_source.append(result.source)
        summary.record(result)

        if (idx + 1) % config.checkpoint_every == 0:
            checkpoint_store.save(idx + 1, summary)

    working["original_pub_year"] = results_year
    working["pub_year_confidence"] = results_conf
    working["original_pub_source"] = results_source
    checkpoint_store.save(len(working), summary)
    return EnrichmentResult(dataframe=working, summary=summary)


def enrich_parquet(config: EnrichmentConfig) -> Summary:
    """
    Load parquet, enrich publication years, write output parquet, and emit a JSON
    summary to stdout.

    Purpose:
        CLI-friendly wrapper around `enrich_dataframe` that handles I/O and logging.

    Args:
        config (EnrichmentConfig): Pre-parsed configuration for paths and tuning.

    Returns:
        Summary: Metrics captured during enrichment.

    Side Effects:
        Reads and writes parquet files, writes cache/checkpoint files, and prints
        JSON to stdout.
    """

    df = pd.read_parquet(config.input_path)  # type: ignore[reportUnknownMemberType]
    fallback_client: FallbackClient = NoopFallback()
    result = enrich_dataframe(
        df,
        config=config,
        client=OpenLibraryClient(
            rate_limit=config.rate_limit,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_initial=config.backoff_initial,
            backoff_cap=config.backoff_cap,
        ),
        cache=FileCache(config.cache_dir),
        checkpoint=FileCheckpoint(config.checkpoint_path),
        fallback=fallback_client,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_parquet(  # type: ignore[reportUnknownMemberType]
        config.output_path, index=False
    )
    print(
        json.dumps(
            {
                "output_path": str(config.output_path),
                "summary": result.summary.to_dict(),
                "rows": len(result.dataframe),
            }
        )
    )
    return result.summary

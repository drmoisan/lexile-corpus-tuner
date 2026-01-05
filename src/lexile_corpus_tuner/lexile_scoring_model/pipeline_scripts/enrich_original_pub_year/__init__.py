from __future__ import annotations

import argparse
import json
import math
import numbers
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .cache_store import CacheStore
from .checkpoint_store import CheckpointStore
from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    OPEN_LIBRARY_URL,
)
from .enrichment_config import EnrichmentConfig
from .enrichment_result import EnrichmentResult
from .fallback_client import FallbackClient
from .file_cache import FileCache
from .file_checkpoint import FileCheckpoint
from .http_client import HttpClient
from .match_candidate import MatchCandidate
from .match_result import Confidence, MatchResult
from .noop_fallback import NoopFallback
from .open_library_client import OpenLibraryClient
from .summary import Summary

if TYPE_CHECKING:
    from collections.abc import Iterable


def normalize_text(value: str) -> str:
    """
    Normalize text for stable matching by removing punctuation and collapsing space.

    Purpose:
        Provide deterministic normalization so cache keys and similarity checks are
        consistent across runs.

    Args:
        value (str): Raw title or author value from the dataset.

    Returns:
        str: Lowercased, punctuation-stripped, whitespace-collapsed string.

    Raises:
        None

    Side Effects:
        None.
    """
    cleaned = re.sub(r"[^\w\s]", " ", value).lower()
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def _is_missing_year(value: object) -> bool:
    """
    Detect missing year values, treating NaN-like values as absent but keeping zeros.

    Purpose:
        Distinguish truly missing data from valid numeric values so enrichment does
        not overwrite legitimate zeros.

    Args:
        value (object): Original value from the dataframe.

    Returns:
        bool: True when the value should be considered missing.

    Raises:
        None

    Side Effects:
        None.
    """
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return math.isnan(float(value))
    return False


def _similarity(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity between two normalized strings.

    Purpose:
        Score overlap between candidate and target strings for fuzzy matching.

    Args:
        a (str): Normalized string to compare.
        b (str): Normalized string to compare.

    Returns:
        float: Jaccard similarity in the range [0.0, 1.0].

    Raises:
        None

    Side Effects:
        None.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def select_best_match(
    *,
    candidates: Iterable[MatchCandidate],
    normalized_title: str,
    normalized_author: str,
    threshold: float,
    disable_fuzzy: bool,
) -> MatchResult:
    """
    Choose the strongest candidate, preferring exact matches then fuzzy scoring.

    Purpose:
        Encapsulate match selection rules so enrichment can remain deterministic and
        testable.

    Args:
        candidates (Iterable[MatchCandidate]): Provider results to evaluate.
        normalized_title (str): Normalized target title.
        normalized_author (str): Normalized target author list.
        threshold (float): Minimum similarity required for fuzzy acceptance.
        disable_fuzzy (bool): When True, only exact matches are eligible.

    Returns:
        MatchResult: Selected year with confidence and source metadata.

    Raises:
        None

    Side Effects:
        None.
    """

    best: MatchCandidate | None = None
    best_score = 0.0
    # Evaluate each provider candidate, preferring exact matches before fuzzy ones.
    for candidate in candidates:
        cand_title = normalize_text(candidate.title)
        cand_author = normalize_text(candidate.author)
        exact_title = cand_title == normalized_title and cand_title != ""
        author_overlap = (
            bool(set(cand_author.split()) & set(normalized_author.split()))
            if normalized_author
            else False
        )
        # Short-circuit when title matches exactly and authors overlap with a year.
        if exact_title and author_overlap and candidate.year is not None:
            return MatchResult(
                year=candidate.year, confidence="high", source=candidate.source
            )

        if disable_fuzzy:
            continue

        title_score = _similarity(normalized_title, cand_title)
        author_score = _similarity(normalized_author, cand_author)
        score = (title_score + author_score) / 2.0
        # Track the best-scoring fuzzy candidate above the acceptance threshold.
        if score >= threshold and score >= best_score and candidate.year is not None:
            best_score = score
            best = MatchCandidate(
                title=candidate.title,
                author=candidate.author,
                year=candidate.year,
                source=candidate.source,
                score=score,
            )

    if best is not None:
        return MatchResult(year=best.year, confidence="low", source=best.source)
    return MatchResult(year=None, confidence="none", source=None)


def enrich_dataframe(
    df: pd.DataFrame,
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

    Raises:
        Exception: Propagates errors from HTTP requests, cache/checkpoint I/O, or
        fallback lookups.
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

    # Walk each row, resuming from any prior checkpoint to avoid reprocessing.
    for idx, row in enumerate(working.itertuples(index=False), start=0):
        if idx < start_index:
            # Preserve previously processed rows by copying stored values.
            existing_year = getattr(row, "original_pub_year", None)
            is_missing_year = _is_missing_year(existing_year)
            results_year.append(None if is_missing_year else existing_year)
            results_conf.append(getattr(row, "pub_year_confidence", "none"))
            results_source.append(getattr(row, "original_pub_source", None))
            continue

        title = getattr(row, "title", "") or ""
        authors = getattr(row, "authors", "") or ""
        normalized_title = normalize_text(title)
        normalized_author = normalize_text(authors)

        cache_key = f"{normalized_title}__{normalized_author}"
        # Prefer cache hits to avoid unnecessary network requests.
        cached = cache_store.get(cache_key)
        if cached is not None:
            result = cached
        else:
            result = select_best_match(
                candidates=client.search(title, authors),
                normalized_title=normalized_title,
                normalized_author=normalized_author,
                threshold=config.fuzzy_threshold,
                disable_fuzzy=config.disable_fuzzy,
            )
            if result.confidence == "none":
                # Try secondary catalogs only when primary matching yields nothing.
                fallback_candidates = fallback_client.search(title, authors)
                result = select_best_match(
                    candidates=fallback_candidates,
                    normalized_title=normalized_title,
                    normalized_author=normalized_author,
                    threshold=config.fuzzy_threshold,
                    disable_fuzzy=config.disable_fuzzy,
                )
            cache_store.set(cache_key, result)

        results_year.append(result.year)
        results_conf.append(result.confidence)
        results_source.append(result.source)
        summary.record(result)

        if (idx + 1) % config.checkpoint_every == 0:
            # Persist progress at configured cadence to enable resumable runs.
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

    Raises:
        Exception: Propagates errors from `enrich_dataframe` or file operations.
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


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser with defaults matching `EnrichmentConfig`.

    Purpose:
        Centralize CLI definitions for reuse in tests and entry points.

    Returns:
        argparse.ArgumentParser: Parser ready to consume CLI arguments.

    Raises:
        None

    Side Effects:
        None.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Enrich Gutenberg parquet with original publication year "
            "using Open Library and fallbacks."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Input parquet path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output parquet path (default: data/meta/gutenberg_books_enhanced.parquet)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path for resumable progress",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=5.0, help="Requests per second"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Max retries per request"
    )
    parser.add_argument(
        "--backoff-initial", type=float, default=0.5, help="Initial backoff seconds"
    )
    parser.add_argument(
        "--backoff-cap", type=float, default=8.0, help="Maximum backoff seconds"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for fuzzy matches",
    )
    parser.add_argument(
        "--disable-fuzzy", action="store_true", help="Require exact matches only"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache directory"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500, help="Checkpoint frequency in rows"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=10.0, help="HTTP timeout per request"
    )
    parser.add_argument(
        "--enable-wikidata", action="store_true", help="Enable Wikidata fallback lookup"
    )
    parser.add_argument(
        "--enable-loc", action="store_true", help="Enable LOC fallback lookup"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> EnrichmentConfig:
    """
    Parse CLI arguments into an `EnrichmentConfig` instance.

    Purpose:
        Convert raw CLI input into a strongly typed configuration.

    Args:
        argv (list[str] | None): Argument list override for testing.

    Returns:
        EnrichmentConfig: Parsed and validated configuration object.

    Raises:
        SystemExit: When required arguments are missing or invalid.

    Side Effects:
        Reads process arguments when `argv` is None.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return EnrichmentConfig(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        backoff_initial=args.backoff_initial,
        backoff_cap=args.backoff_cap,
        fuzzy_threshold=args.fuzzy_threshold,
        disable_fuzzy=args.disable_fuzzy,
        checkpoint_every=args.checkpoint_every,
        timeout_seconds=args.timeout_seconds,
        enable_wikidata=args.enable_wikidata,
        enable_loc=args.enable_loc,
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point; run enrichment and return a shell-friendly exit code for
    schedulers.

    Purpose:
        Provide a simple executable boundary for schedulers or manual invocation.

    Args:
        argv (list[str] | None): Optional argument override for testing.

    Returns:
        int: Zero on success; non-zero on failure.

    Side Effects:
        Executes enrichment, performing I/O and network requests.

    Raises:
        None (returns exit codes instead of propagating errors within this boundary).
    """

    config = parse_args(argv)
    try:
        enrich_parquet(config)
        return 0
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Enrichment failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "CacheStore",
    "CheckpointStore",
    "Confidence",
    "EnrichmentConfig",
    "EnrichmentResult",
    "FallbackClient",
    "FileCache",
    "FileCheckpoint",
    "HttpClient",
    "MatchCandidate",
    "MatchResult",
    "NoopFallback",
    "OpenLibraryClient",
    "Summary",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_OUTPUT",
    "OPEN_LIBRARY_URL",
    "normalize_text",
    "select_best_match",
    "enrich_dataframe",
    "enrich_parquet",
    "build_arg_parser",
    "parse_args",
    "main",
]

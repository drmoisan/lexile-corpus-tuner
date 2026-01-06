"""
CLI helpers for the original publication year enrichment pipeline.

Purpose:
    Host argument parsing and the CLI entrypoint separately from package
    initialization to keep imports lightweight while preserving the public API.

Usage:
    from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts \
        import enrich_original_pub_year as enrich
    config = enrich.cli.parse_args()
    exit_code = enrich.cli.main()

Flow:
    - build_arg_parser defines CLI options
    - parse_args produces an EnrichmentConfig
    - main orchestrates the CLI boundary and returns an exit code

Invariants / Constraints:
    The CLI helpers mirror the previous package-level signatures for backward
    compatibility.

Side Effects:
    Reads process arguments, performs I/O through downstream calls to enrichment
    routines, and prints errors on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import DEFAULT_CACHE_DIR, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT
from .enricher import enrich_parquet
from .enrichment_config import EnrichmentConfig


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
            "using Open Library."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Input parquet path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output parquet path (default: "
            "data/meta/gutenberg/gutenberg_books_enhanced.parquet)"
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
        max_retries=args.max_retries,
        backoff_initial=args.backoff_initial,
        backoff_cap=args.backoff_cap,
        fuzzy_threshold=args.fuzzy_threshold,
        disable_fuzzy=args.disable_fuzzy,
        checkpoint_every=args.checkpoint_every,
        timeout_seconds=args.timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point; run enrichment and return a shell-friendly exit code.

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

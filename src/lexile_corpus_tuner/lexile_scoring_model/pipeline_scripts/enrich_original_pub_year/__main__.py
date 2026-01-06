"""
Module entry point for running publication-year enrichment as a package.

Purpose:
    Provide a python -m target that delegates to the CLI main function. This
    keeps the executable surface aligned with the package structure while
    preserving the existing runtime behavior.

Usage:
    python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.
    enrich_original_pub_year \
        --input data/meta/gutenberg/gutenberg_books.parquet \
        --output data/meta/gutenberg/gutenberg_books_enhanced.parquet

Flow:
    Import the CLI main function and execute it, returning its exit code.

Invariants / Constraints:
    Expects the CLI main function to return an integer exit code consistent
    with standard CLI conventions.

Side Effects:
    Whatever side effects are produced by the delegated main call (I/O, network).
"""

from __future__ import annotations

from .cli import main as _cli_main


def _run() -> int:
    """
    Execute the CLI main function and propagate its exit code.

    Purpose:
        Provide a small wrapper so python -m execution can raise SystemExit with
        the package's exit code, matching common CLI expectations.

    Args:
        None

    Returns:
        int: Exit code emitted by the CLI main function.

    Raises:
        None directly; any exceptions are surfaced through the delegated main
        function's behavior.

    Side Effects:
        Mirrors the side effects of the CLI main function (I/O, network).
    """

    return _cli_main()


if __name__ == "__main__":
    raise SystemExit(_run())

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import DEFAULT_CACHE_DIR, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class EnrichmentConfig:
    """
    Configuration for publication-year enrichment, covering I/O paths and tuning knobs.

    Purpose:
        Capture all runtime controls for the enrichment pipeline so callers can
        construct configurations programmatically or via CLI parsing.

    Usage:
        Pass an instance to `enrich_dataframe` or `enrich_parquet` to steer cache
        locations, checkpointing cadence, HTTP behavior, and fuzzy matching.

    Flow:
        Values are consumed by cache, checkpoint, HTTP, and matching components
        without further mutation once set.

    Invariants / Constraints:
        Paths must be writable when enrichment is expected to persist cache or
        checkpoints. Rate limit and backoff values must be positive to avoid
        zero-division or runaway retry behavior.

    Attributes:
        input_path (Path): Source parquet to enrich.
        output_path (Path): Destination parquet path for enriched results.
        checkpoint_path (Path): Location for resumable progress markers.
         cache_dir (Path): Directory for cached match results.
         rate_limit (float): Requests per second allowed against the HTTP source.
         max_retries (int): Maximum retry attempts per HTTP request.
         backoff_initial (float): Base delay before exponential backoff.
         backoff_cap (float): Maximum delay between retries.
         fuzzy_threshold (float): Minimum similarity for fuzzy matches to be accepted.
         disable_fuzzy (bool): When True, only exact matches are considered valid.
         timeout_seconds (float): HTTP timeout per request.
         checkpoint_every (int): Frequency (rows) for writing checkpoints.
    """

    input_path: Path
    output_path: Path = DEFAULT_OUTPUT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    cache_dir: Path = DEFAULT_CACHE_DIR
    rate_limit: float = 5.0
    max_retries: int = 5
    backoff_initial: float = 0.5
    backoff_cap: float = 8.0
    fuzzy_threshold: float = 0.9
    disable_fuzzy: bool = False
    timeout_seconds: float = 10.0
    checkpoint_every: int = 500

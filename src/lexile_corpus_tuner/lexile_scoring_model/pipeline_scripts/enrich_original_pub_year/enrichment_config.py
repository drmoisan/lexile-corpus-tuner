from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import DEFAULT_CACHE_DIR, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class EnrichmentConfig:
    """
    Configuration for enrichment behavior, including rate limits, retries, fuzziness,
    checkpointing, and I/O paths.
    """

    input_path: Path
    output_path: Path = DEFAULT_OUTPUT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    cache_dir: Path = DEFAULT_CACHE_DIR
    rate_limit: float = 5.0
    batch_size: int = 50
    max_retries: int = 5
    backoff_initial: float = 0.5
    backoff_cap: float = 8.0
    fuzzy_threshold: float = 0.9
    disable_fuzzy: bool = False
    timeout_seconds: float = 10.0
    checkpoint_every: int = 500
    enable_wikidata: bool = False
    enable_loc: bool = False

from __future__ import annotations

from pathlib import Path

from .cache_store import CacheStore
from .checkpoint_store import CheckpointStore
from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    OPEN_LIBRARY_URL,
)
from .enricher import enrich_dataframe, enrich_parquet
from .enrichment_config import EnrichmentConfig
from .enrichment_result import EnrichmentResult
from .fallback_client import FallbackClient
from .file_cache import FileCache
from .file_checkpoint import FileCheckpoint
from .http_client import HttpClient
from .match_candidate import MatchCandidate
from .match_result import Confidence, MatchResult
from .match_utils import normalize_text, select_best_match
from .noop_fallback import NoopFallback
from .open_library_client import OpenLibraryClient
from .summary import Summary

_CLI_EXPORTS = {"build_arg_parser", "parse_args", "main"}


def __getattr__(name: str):
    """
    Lazily expose CLI helpers from the dedicated cli module to preserve API surface.

    Purpose:
        Avoid eager CLI imports during package initialization while maintaining
        backward compatibility for callers that previously accessed CLI helpers
        directly from the package.

    Args:
        name (str): Attribute name requested by the caller.

    Returns:
        object: Attribute fetched from the cli module when supported.

    Raises:
        AttributeError: When the requested name is not one of the CLI exports.

    Side Effects:
        Imports the cli module on first access to a CLI attribute.
    """

    if name in _CLI_EXPORTS:
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!s} has no attribute {name!s}")


def __dir__() -> list[str]:
    """
    Provide a complete attribute list including lazily exported CLI helpers.

    Returns:
        list[str]: Sorted list of attribute names for introspection tools.
    """

    return sorted(set(globals()) | _CLI_EXPORTS)


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
    "Path",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from .summary import Summary


@dataclass
class EnrichmentResult:
    """
    Aggregate result produced by enrichment, pairing the dataframe with metrics.

    Purpose:
        Capture both the enriched dataset and the processing summary so callers can
        persist data while also reporting coverage and error counts.

    Usage:
        Returned from `enrich_dataframe` and `enrich_parquet`, then consumed by CLI
        output or downstream pipelines.

    Attributes:
        dataframe (pd.DataFrame): Copy of the source data with enrichment columns
        added.
        summary (Summary): Counts of high/low/none matches and errors for reporting
        or checkpoints.
    """

    dataframe: pd.DataFrame
    summary: Summary

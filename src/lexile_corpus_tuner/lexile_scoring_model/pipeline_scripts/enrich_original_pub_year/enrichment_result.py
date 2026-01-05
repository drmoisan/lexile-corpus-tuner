from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from .summary import Summary


@dataclass
class EnrichmentResult:
    """Container for the enriched dataframe and the summary metrics."""

    dataframe: pd.DataFrame
    summary: Summary

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class WordFrequency:
    count: int
    freq_per_5m: float
    log_freq_per_5m: float
    rank: int


def load_frequency_table(path: Path | None = None) -> Dict[str, WordFrequency]:
    """
    Load a TSV table mapping tokens to WordFrequency values.

    Parameters
    ----------
    path:
        Custom path to the TSV. Defaults to data/freq/word_frequencies.tsv.
    """
    if path is None:
        path = Path("data/freq/word_frequencies.tsv")

    table: Dict[str, WordFrequency] = {}
    if not path.exists():
        return table

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            token = row["token"]
            table[token] = WordFrequency(
                count=int(row["count"]),
                freq_per_5m=float(row["freq_per_5m"]),
                log_freq_per_5m=float(row["log_freq_per_5m"]),
                rank=int(row["rank"]),
            )

    return table

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

SHARDS_ROOT = Path("data/corpus/normalized/shards")
FREQ_ROOT = Path("data/freq")
FREQ_TSV = FREQ_ROOT / "word_frequencies.tsv"
FREQ_META = FREQ_ROOT / "word_frequencies.meta.json"
CORPUS_META_PATH = Path("data/meta/corpus_sources.json")

LOGGER = logging.getLogger(__name__)


def compute_global_frequencies() -> None:
    """Compute global token counts + log frequencies over normalized shards."""
    raw_counts: Counter[str] = Counter()
    weighted_counts: dict[str, float] = defaultdict(float)
    source_weights = _load_source_weights()
    weighted_total_tokens = 0.0
    shard_files = sorted(SHARDS_ROOT.glob("*.jsonl"))
    if not shard_files:
        LOGGER.warning(
            "No shard files found under %s. Run corpus normalize first.", SHARDS_ROOT
        )
        return

    for shard_path in shard_files:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = record.get("tokens", [])
                source_id = str(record.get("source_id", "unknown"))
                weight = source_weights.get(source_id, 1.0)
                raw_counts.update(tokens)
                for token in tokens:
                    weighted_counts[token] += weight
                weighted_total_tokens += len(tokens) * weight

    total_tokens = sum(raw_counts.values())
    if weighted_total_tokens == 0:
        LOGGER.warning("Shard files contained zero tokens; skipping frequency write.")
        return

    FREQ_ROOT.mkdir(parents=True, exist_ok=True)
    with FREQ_TSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["token", "count", "freq_per_5m", "log_freq_per_5m", "rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        sorted_tokens = sorted(
            weighted_counts.items(), key=lambda item: item[1], reverse=True
        )
        for rank, (token, weighted_count) in enumerate(sorted_tokens, start=1):
            freq_per_5m = weighted_count * 5_000_000.0 / weighted_total_tokens
            log_freq = math.log(freq_per_5m + 1e-12)
            writer.writerow(
                {
                    "token": token,
                    "count": raw_counts[token],
                    "freq_per_5m": f"{freq_per_5m:.9f}",
                    "log_freq_per_5m": f"{log_freq:.12f}",
                    "rank": rank,
                }
            )

    meta = {
        "version": _current_version(),
        "total_tokens": total_tokens,
        "weighted_total_tokens": weighted_total_tokens,
        "num_types": len(weighted_counts),
        "source_meta_path": str(CORPUS_META_PATH),
        "notes": "Computed from normalized shards; frequencies are per 5M tokens.",
    }
    FREQ_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info(
        "Computed global frequencies for %d tokens (%d types).",
        total_tokens,
        len(weighted_counts),
    )


def _current_version() -> str:
    from datetime import date

    return date.today().isoformat()


def _load_source_weights() -> dict[str, float]:
    if not CORPUS_META_PATH.exists():
        return {}
    try:
        data = json.loads(CORPUS_META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    weights: dict[str, float] = {}
    for entry in data.get("sources", []):
        source_id = entry.get("id")
        weight = entry.get("weight", 1.0)
        if isinstance(source_id, str):
            try:
                weights[source_id] = float(weight)
            except (TypeError, ValueError):
                weights[source_id] = 1.0
    return weights

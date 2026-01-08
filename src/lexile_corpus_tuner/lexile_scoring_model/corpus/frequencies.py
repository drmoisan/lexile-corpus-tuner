from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict, cast

from lexile_corpus_tuner.lexile_scoring_model.corpus.schema import NormalizedDocument

SHARDS_ROOT = Path("data/corpus/normalized/shards")
FREQ_ROOT = Path("data/freq")
FREQ_TSV = FREQ_ROOT / "word_frequencies.tsv"
FREQ_META = FREQ_ROOT / "word_frequencies.meta.json"
WEIGHTED_FREQ_TSV = FREQ_ROOT / "weighted_word_frequencies.tsv"
WEIGHTED_FREQ_META = FREQ_ROOT / "weighted_word_frequencies.meta.json"
CORPUS_META_PATH = Path("resources/meta/corpus_sources.json")
CORPUS_STATS_PATH = FREQ_ROOT / "corpus_stats.json"

LOGGER = logging.getLogger(__name__)


def compute_global_frequencies(
    weighted: bool = False, config_path: Path | None = None
) -> None:
    """Compute global token counts + log frequencies over normalized shards."""
    raw_counts: Counter[str] = Counter()
    weighted_counts: dict[str, float] = defaultdict(float)
    token_stats = _init_stats()
    weight_matrix = _load_weight_matrix(config_path) if weighted else {}
    shard_files = sorted(SHARDS_ROOT.glob("*.jsonl"))
    if not shard_files:
        LOGGER.warning(
            "No shard files found under %s. Run corpus normalize first.", SHARDS_ROOT
        )
        return

    weighted_total_tokens = 0.0
    for shard_path in shard_files:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    doc = NormalizedDocument.from_json(record)
                except (json.JSONDecodeError, ValueError):
                    continue

                tokens = doc.tokens
                doc_weight = (
                    _resolve_doc_weight(doc, weight_matrix) if weighted else 1.0
                )
                raw_counts.update(tokens)
                _accumulate_stats(token_stats, doc, len(tokens), doc_weight)
                for token in tokens:
                    weighted_counts[token] += doc_weight
                weighted_total_tokens += len(tokens) * doc_weight

    total_tokens = sum(raw_counts.values())
    if weighted_total_tokens == 0:
        LOGGER.warning("Shard files contained zero tokens; skipping frequency write.")
        return

    output_tsv = WEIGHTED_FREQ_TSV if weighted else FREQ_TSV
    output_meta = WEIGHTED_FREQ_META if weighted else FREQ_META
    FREQ_ROOT.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
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
        "weighted": weighted,
        "weight_config_path": str(config_path) if config_path else None,
        "notes": "Computed from normalized shards; frequencies are per 5M tokens.",
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_stats_report(token_stats, weighted)
    LOGGER.info(
        "Computed %sfrequencies for %d tokens (%d types).",
        "weighted " if weighted else "",
        total_tokens,
        len(weighted_counts),
    )


def _current_version() -> str:
    from datetime import date

    return date.today().isoformat()


def _load_source_weights() -> dict[str, float]:  # pyright: ignore[reportUnusedFunction]
    if not CORPUS_META_PATH.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(CORPUS_META_PATH.read_text(encoding="utf-8"))
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


def _load_weight_matrix(config_path: Path | None) -> dict[str, dict[str, float]]:
    if config_path is None:
        return {}
    if not config_path.exists():
        LOGGER.warning("Weight config not found at %s; defaulting to 1.0.", config_path)
        return {}
    loaded: Any
    try:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore

            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load weight config %s: %s", config_path, exc)
        return {}

    if not isinstance(loaded, dict):
        return {}

    loaded_dict: dict[Any, Any] = cast(dict[Any, Any], loaded)
    normalized_loaded: dict[str, Any] = {
        str(key): value for key, value in loaded_dict.items()
    }
    weights_section_any = normalized_loaded.get("weights")
    if not isinstance(weights_section_any, dict):
        return {}

    weights_section: dict[str, Any] = cast(dict[str, Any], weights_section_any)
    matrix: dict[str, dict[str, float]] = {}
    for source_id, value in weights_section.items():
        if not isinstance(value, dict):
            continue
        value_dict: dict[Any, Any] = cast(dict[Any, Any], value)
        normalized: dict[str, float] = {}
        for era, weight in value_dict.items():
            era_key = str(era)
            try:
                normalized[era_key] = float(weight)
            except (TypeError, ValueError):
                continue
        matrix[source_id] = normalized
    return matrix


class Stats(TypedDict):
    raw_tokens: defaultdict[str, defaultdict[str, int]]
    weighted_tokens: defaultdict[str, defaultdict[str, float]]


def _resolve_doc_weight(
    doc: NormalizedDocument, matrix: dict[str, dict[str, float]]
) -> float:
    if doc.weight is not None:
        try:
            return float(doc.weight)
        except (TypeError, ValueError):
            return 1.0
    source_map = matrix.get(doc.source_id, {})
    if doc.era_bucket in source_map:
        return source_map[doc.era_bucket]
    if "default" in source_map:
        return source_map["default"]
    return 1.0


def _init_stats() -> Stats:
    raw_tokens: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    weighted_tokens: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    return {
        "raw_tokens": raw_tokens,
        "weighted_tokens": weighted_tokens,
    }


def _accumulate_stats(
    stats: Stats, doc: NormalizedDocument, num_tokens: int, weight: float
) -> None:
    stats["raw_tokens"][doc.source_id][doc.era_bucket] += num_tokens
    stats["weighted_tokens"][doc.source_id][doc.era_bucket] += num_tokens * weight


def _write_stats_report(stats: Stats, weighted: bool) -> None:
    report = {
        "weighted": weighted,
        "raw_tokens": _dictify(stats["raw_tokens"]),
        "weighted_tokens": _dictify(stats["weighted_tokens"]),
    }
    CORPUS_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_STATS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _dictify(
    value: defaultdict[str, Any] | dict[str, Any] | float | int,
) -> dict[str, Any] | float | int:
    if isinstance(value, defaultdict):
        return {str(k): _dictify(v) for k, v in value.items()}
    return value

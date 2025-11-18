from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, List, Tuple

from ..textutils import iter_tokens, normalize_text

RAW_ROOT = Path("data/corpus/raw")
NORMALIZED_ROOT = Path("data/corpus/normalized")
SHARDS_ROOT = NORMALIZED_ROOT / "shards"
SUMMARY_PATH = NORMALIZED_ROOT / "normalized_summary.json"

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class NormalizedShardMeta:
    shard_id: str
    source_id: str
    num_tokens: int
    num_texts: int


def normalize_all_sources(
    shard_size_tokens: int = 100_000,
) -> List[NormalizedShardMeta]:
    """Normalize and tokenize raw sources into fixed-size shards."""
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    SHARDS_ROOT.mkdir(parents=True, exist_ok=True)

    shard_metas: list[NormalizedShardMeta] = []
    current_records: list[dict[str, Any]] = []
    current_token_count = 0
    current_source_id: str | None = None
    shard_index = 1

    for source_id, text_id, raw_text in iter_raw_texts():
        normalized = normalize_text(raw_text)
        tokens = list(iter_tokens(normalized))
        if not tokens:
            continue

        record = {"source_id": source_id, "text_id": text_id, "tokens": tokens}

        if current_source_id is None:
            current_source_id = source_id

        if source_id != current_source_id and current_records:
            shard_metas.append(
                _write_shard(shard_index, current_source_id, current_records)
            )
            shard_index += 1
            current_records = []
            current_token_count = 0
            current_source_id = source_id

        current_records.append(record)
        current_token_count += len(tokens)

        if current_token_count >= shard_size_tokens:
            shard_metas.append(
                _write_shard(shard_index, current_source_id, current_records)
            )
            shard_index += 1
            current_records = []
            current_token_count = 0

    if current_records and current_source_id:
        shard_metas.append(
            _write_shard(shard_index, current_source_id, current_records)
        )

    _write_summary(shard_metas)
    LOGGER.info("Wrote %d normalized shards", len(shard_metas))
    return shard_metas


def iter_raw_texts() -> Iterator[Tuple[str, str, str]]:
    """Yield (source_id, text_id, raw_text) tuples for every available source."""
    yield from _iter_gutenberg_texts()
    yield from _iter_simple_wiki_texts()


def _iter_gutenberg_texts() -> Iterator[Tuple[str, str, str]]:
    base_dir = RAW_ROOT / "gutenberg"
    if not base_dir.exists():
        return
    for path in sorted(base_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        source_id = _classify_gutenberg_path(path)
        text_id = f"gutenberg-{path.stem}"
        yield source_id, text_id, text


def _iter_simple_wiki_texts() -> Iterator[Tuple[str, str, str]]:
    base_dir = RAW_ROOT / "simple_wiki"
    if not base_dir.exists():
        return

    for path in sorted(base_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text_id = f"simple_wiki-{path.stem}"
        yield "simple_wiki", text_id, text

    for path in sorted(base_dir.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text") or record.get("content")
                    if not text:
                        continue
                    text_id = record.get("id") or f"{path.stem}-{idx}"
                    yield "simple_wiki", f"simple_wiki-{text_id}", str(text)
        except OSError:
            continue


def _classify_gutenberg_path(path: Path) -> str:
    name = path.stem.lower()
    path_parts = [p.lower() for p in path.parts]
    child_markers = ("child", "juvenile", "children", "kid", "ya")
    if any(marker in name for marker in child_markers):
        return "gutenberg_child"
    if any(marker in part for part in path_parts for marker in child_markers):
        return "gutenberg_child"
    return "gutenberg_other"


def _write_shard(
    shard_index: int, source_id: str, records: list[dict[str, Any]]
) -> NormalizedShardMeta:
    shard_id = f"shard-{shard_index:06d}-{source_id}"
    shard_path = SHARDS_ROOT / f"{shard_id}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    num_tokens = sum(len(record["tokens"]) for record in records)
    with shard_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")

    meta = NormalizedShardMeta(
        shard_id=shard_id,
        source_id=source_id,
        num_tokens=num_tokens,
        num_texts=len(records),
    )
    return meta


def _write_summary(shards: list[NormalizedShardMeta]) -> None:
    summary = {
        "version": _dt.date.today().isoformat(),
        "num_shards": len(shards),
        "shards": [asdict(shard) for shard in shards],
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

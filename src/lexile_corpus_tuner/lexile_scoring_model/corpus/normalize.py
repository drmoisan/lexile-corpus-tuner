from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lexile_corpus_tuner.corpus_tuning_pipeline.textutils import (
    iter_tokens,
    normalize_text,
)
from lexile_corpus_tuner.lexile_scoring_model.corpus.schema import NormalizedDocument

if TYPE_CHECKING:
    from collections.abc import Iterator

RAW_ROOT = Path("data/corpus/raw")
NORMALIZED_ROOT = Path("data/corpus/normalized")
SHARDS_ROOT = NORMALIZED_ROOT / "shards"
SUMMARY_PATH = NORMALIZED_ROOT / "normalized_summary.json"

MIN_DOC_TOKENS = 1_000
MAX_DOC_TOKENS = 3_000
WIKI_MIN_TOKENS = 150
WIKI_MAX_TOKENS = 8_000
CURRENT_YEAR = _dt.date.today().year

LOGGER = logging.getLogger(__name__)
OER_SOURCE_DIRS = ("openstax", "ck12")


@dataclass(slots=True)
class NormalizedShardMeta:
    shard_id: str
    source_id: str
    num_tokens: int
    num_texts: int


@dataclass(slots=True)
class RawDocument:
    source_id: str
    text_id: str
    text: str
    path: Path | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class DocumentMeta:
    genre: str
    publication_year: int | None
    era_bucket: str
    intended_audience: str
    grade_band: str | None


def normalize_all_sources(
    shard_size_tokens: int = 100_000,
    allowed_sources: set[str] | None = None,
) -> list[NormalizedShardMeta]:
    """Normalize and tokenize raw sources into fixed-size shards."""
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    SHARDS_ROOT.mkdir(parents=True, exist_ok=True)

    shard_metas: list[NormalizedShardMeta] = []
    current_records: list[dict[str, Any]] = []
    current_token_count = 0
    current_source_id: str | None = None
    shard_index = 1

    for raw_doc in iter_raw_texts(allowed_sources=allowed_sources):
        normalized = normalize_text(raw_doc.text)
        tokens = list(iter_tokens(normalized))
        meta = _build_metadata(raw_doc, tokens)
        if not _passes_filters(raw_doc, tokens, meta):
            continue
        chunks = _chunk_tokens(
            tokens, min_tokens=MIN_DOC_TOKENS, max_tokens=MAX_DOC_TOKENS
        )
        if not chunks:
            continue

        if current_source_id is None:
            current_source_id = raw_doc.source_id

        if raw_doc.source_id != current_source_id and current_records:
            shard_metas.append(
                _write_shard(shard_index, current_source_id, current_records)
            )
            shard_index += 1
            current_records = []
            current_token_count = 0
            current_source_id = raw_doc.source_id

        for chunk_index, chunk_tokens in enumerate(chunks):
            chunk_text_id = f"{raw_doc.text_id}-chunk-{chunk_index:04d}"
            record = {
                "source_id": raw_doc.source_id,
                "text_id": chunk_text_id,
                "tokens": chunk_tokens,
                "genre": meta.genre,
                "publication_year": meta.publication_year,
                "era_bucket": meta.era_bucket,
                "intended_audience": meta.intended_audience,
                "grade_band": meta.grade_band,
                "weight": None,
            }
            _apply_metadata_fallbacks(record)
            if not _is_valid_record(record):
                continue
            current_records.append(record)
            current_token_count += len(chunk_tokens)

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


def iter_raw_texts(allowed_sources: set[str] | None = None) -> Iterator[RawDocument]:
    """Yield normalized raw documents from available sources, optionally filtered."""
    allowed = (
        {source.lower() for source in allowed_sources} if allowed_sources else None
    )
    if allowed is None or "gutenberg" in allowed or "gutenberg_child" in allowed:
        yield from _iter_gutenberg_texts()
    if allowed is None or "simple_wiki" in allowed or "standard_wiki" in allowed:
        yield from _iter_simple_wiki_texts()
    if allowed is None or any(source in allowed for source in OER_SOURCE_DIRS):
        yield from _iter_oer_texts()


def _iter_gutenberg_texts() -> Iterator[RawDocument]:
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
        yield RawDocument(source_id=source_id, text_id=text_id, text=text, path=path)


def _iter_simple_wiki_texts() -> Iterator[RawDocument]:
    base_dir = RAW_ROOT / "simple_wiki"
    if not base_dir.exists():
        return

    for path in sorted(base_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text_id = f"simple_wiki-{path.stem}"
        yield RawDocument(
            source_id="simple_wiki", text_id=text_id, text=text, path=path
        )

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
                    yield RawDocument(
                        source_id="simple_wiki",
                        text_id=f"simple_wiki-{text_id}",
                        text=str(text),
                        path=path,
                    )
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


def _build_metadata(raw_doc: RawDocument, tokens: list[str]) -> DocumentMeta:
    if raw_doc.source_id.startswith("gutenberg"):
        return _build_gutenberg_metadata(raw_doc)
    if raw_doc.source_id in {"simple_wiki", "standard_wiki"}:
        return DocumentMeta(
            genre="expository",
            publication_year=CURRENT_YEAR,
            era_bucket="post_2000",
            intended_audience="general",
            grade_band=None,
        )
    if raw_doc.source_id in OER_SOURCE_DIRS:
        grade_band = None
        if raw_doc.extra:
            grade_band = raw_doc.extra.get("grade_band") or raw_doc.extra.get("grade")
            if grade_band is not None:
                grade_band = str(grade_band)
        return DocumentMeta(
            genre="instructional",
            publication_year=CURRENT_YEAR,
            era_bucket="post_2000",
            intended_audience="educational",
            grade_band=grade_band,
        )
    return DocumentMeta(
        genre="expository",
        publication_year=None,
        era_bucket="unknown",
        intended_audience="general",
        grade_band=None,
    )


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


def _iter_oer_texts() -> Iterator[RawDocument]:
    for source in OER_SOURCE_DIRS:
        base_dir = RAW_ROOT / source
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            text_id = f"{source}-{path.stem}"
            yield RawDocument(source_id=source, text_id=text_id, text=text, path=path)
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
                        yield RawDocument(
                            source_id=source,
                            text_id=f"{source}-{text_id}",
                            text=str(text),
                            path=path,
                            extra=record,
                        )
            except OSError:
                continue


def _chunk_tokens(
    tokens: list[str],
    min_tokens: int = MIN_DOC_TOKENS,
    max_tokens: int = MAX_DOC_TOKENS,
) -> list[list[str]]:
    if not tokens:
        return []
    chunks: list[list[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < min_tokens and chunks:
            chunks[-1].extend(chunk)
            break
        chunks.append(chunk)
        start = end
    return chunks


def _build_gutenberg_metadata(raw_doc: RawDocument) -> DocumentMeta:
    publication_year = _extract_first_year(raw_doc.text)
    era_bucket = _infer_era_bucket(publication_year)
    intended_audience = "child" if raw_doc.source_id == "gutenberg_child" else "general"
    return DocumentMeta(
        genre="narrative",
        publication_year=publication_year,
        era_bucket=era_bucket,
        intended_audience=intended_audience,
        grade_band=None,
    )


def _extract_first_year(text: str) -> int | None:
    header = text[:4000]
    match = re.search(r"(1[5-9]\d{2}|20\d{2})", header)
    if not match:
        return None
    year = int(match.group(1))
    if 1500 <= year <= CURRENT_YEAR:
        return year
    return None


def _infer_era_bucket(publication_year: int | None) -> str:
    if publication_year is None:
        return "unknown"
    if publication_year < 1950:
        return "pre_1950"
    if publication_year < 2000:
        return "1950_2000"
    return "post_2000"


def _passes_filters(
    raw_doc: RawDocument, tokens: list[str], meta: DocumentMeta
) -> bool:
    if raw_doc.source_id in {"simple_wiki", "standard_wiki"}:
        return _wiki_passes_filters(raw_doc.text, tokens)
    return bool(tokens)


def _apply_metadata_fallbacks(record: dict[str, Any]) -> None:
    if not record.get("publication_year"):
        record["era_bucket"] = record.get("era_bucket") or "unknown"
    if not record.get("intended_audience"):
        record["intended_audience"] = "general"
    if "weight" not in record:
        record["weight"] = None


def _is_valid_record(record: dict[str, Any]) -> bool:
    try:
        NormalizedDocument.from_json(record)
    except ValueError:
        LOGGER.warning(
            "Skipping record with missing required fields: %s", record.get("text_id")
        )
        return False
    if not record.get("tokens"):
        return False
    return True


def _wiki_passes_filters(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    if not (WIKI_MIN_TOKENS <= len(tokens) <= WIKI_MAX_TOKENS):
        return False
    lowered = text.lower()
    if lowered.startswith("#redirect"):
        return False
    if "{{disambig" in lowered or "disambiguation" in lowered[:200]:
        return False
    if "{{stub" in lowered[:500]:
        return False
    return True

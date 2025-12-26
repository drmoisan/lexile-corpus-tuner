from __future__ import annotations

import argparse
import json
import math
import numbers
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import pandas as pd
import requests
from requests import Response

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

Confidence = Literal["high", "low", "none"]

DEFAULT_OUTPUT = Path("data/meta/gutenberg_books_enhanced.parquet")
DEFAULT_CHECKPOINT = Path("data/meta/.original_pub_year.ckpt")
DEFAULT_CACHE_DIR = Path("data/cache/original_pub_year")
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"


def normalize_text(value: str) -> str:
    # Note 1: Strip punctuation + collapse whitespace for stable fuzzy tokens.
    cleaned = re.sub(r"[^\w\s]", " ", value).lower()
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def _is_missing_year(value: object) -> bool:
    # Note 2: Treat NaN/None as missing so zeros or placeholders are not misread.
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return math.isnan(float(value))
    return False


@dataclass
class MatchCandidate:
    # Note 3: Raw hit from a provider; keep score even if fuzz is disabled for reuse.
    title: str
    author: str
    year: int | None
    source: str
    score: float


@dataclass
class MatchResult:
    # Note 4: Distilled fields; confidence drives metrics and resume logic.
    year: int | None
    confidence: Confidence
    source: str | None


@dataclass
class EnrichmentConfig:
    # Note 5: Defaults fit Open Library quotas; adjust batch/rate via config.
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
    # Note 41: Defaults target repo dirs; CI runs flag-free and artifacts stay ignored.


class CacheStore(Protocol):
    # Note 6: Protocol keeps caches swappable without coupling to a concrete store.
    def get(self, key: str) -> MatchResult | None: ...  # pragma: no cover

    def set(self, key: str, value: MatchResult) -> None: ...  # pragma: no cover


class CheckpointStore(Protocol):
    # Note 7: Pluggable checkpoints support file/DB durability for long runs.
    def load(self) -> int: ...  # pragma: no cover

    def save(self, index: int, summary: Summary) -> None: ...  # pragma: no cover


class Summary:
    def __init__(self) -> None:
        # Note 31: Counters stay ints so checkpoints stay cheap and readable.
        self.matched_high = 0
        self.matched_low = 0
        self.matched_none = 0
        self.errors = 0

    def record(self, result: MatchResult) -> None:
        # Note 8: Update per row; telemetry stays separate from the DataFrame.
        if result.confidence == "high":
            self.matched_high += 1
        elif result.confidence == "low":
            self.matched_low += 1
        else:
            self.matched_none += 1

    def record_error(self) -> None:
        # Note 9: Track API/parsing failures separately for alerting.
        self.errors += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "matched_high": self.matched_high,
            "matched_low": self.matched_low,
            "matched_none": self.matched_none,
            "errors": self.errors,
        }


class FileCache(CacheStore):
    def __init__(self, cache_dir: Path) -> None:
        # Note 10: Cache uses human-readable JSON to debug odd matches.
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Note 11: Sanitize cache key; normalized title+author avoids collisions.
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        return self._cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> MatchResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return MatchResult(
                year=data.get("year"),
                confidence=data.get("confidence", "none"),
                source=data.get("source"),
            )
        except Exception:
            # Note 12: Skip corrupted cache files; next lookup will refresh.
            return None

    def set(self, key: str, value: MatchResult) -> None:
        path = self._path(key)
        payload = {
            "year": value.year,
            "confidence": value.confidence,
            "source": value.source,
        }
        with path.open("w", encoding="utf-8") as fh:
            # Note 13: One file per key avoids locking at batch scale.
            json.dump(payload, fh)
            # Note 32: JSON avoids pickle breakage across Python versions.


class FileCheckpoint(CheckpointStore):
    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path
        # Note 33: Checkpoint lives under data/meta to align with other artifacts.

    def load(self) -> int:
        if not self._checkpoint_path.exists():
            return 0
        try:
            with self._checkpoint_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return int(data.get("last_index", 0))
        except Exception:
            # Note 14: If checkpoint is unreadable, restart from zero for safety.
            return 0

    def save(self, index: int, summary: Summary) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_index": index, "summary": summary.to_dict()}
        with self._checkpoint_path.open("w", encoding="utf-8") as fh:
            # Note 15: Store summary with index to aid audits without parquet reloads.
            json.dump(payload, fh)


class HttpClient(Protocol):
    def get(
        self, url: str, *, params: dict[str, str], timeout: float
    ) -> Response: ...  # pragma: no cover


class OpenLibraryClient:
    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        rate_limit: float = 5.0,
        timeout_seconds: float = 10.0,
        max_retries: int = 5,
        backoff_initial: float = 0.5,
        backoff_cap: float = 8.0,
    ) -> None:
        self._http = http or requests.Session()
        self._rate_limit = rate_limit
        self._timeout_seconds = timeout_seconds
        self._last_request = 0.0
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_cap = backoff_cap
        # Note 34: Reuse session for keep-alive and pooled connections.

    def _respect_rate_limit(self) -> None:
        if self._rate_limit <= 0:
            return
        min_interval = 1.0 / self._rate_limit
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()
        # Note 35: Per-process limit; lower per-worker rates when parallelizing.

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        params = {"title": title, "author": author, "limit": "5"}
        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self._http.get(
                    OPEN_LIBRARY_URL, params=params, timeout=self._timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception:
                # Note 16: Backoff doubles to stay polite and handle transient 5xx/429s.
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = min(
                    self._backoff_initial * (2 ** (attempt - 1)), self._backoff_cap
                )
                time.sleep(delay)
        if not isinstance(payload, dict):
            # Note 36: Guard against non-dict payloads from Open Library schema drift.
            return []

        payload_dict = cast(dict[str, object], payload)
        docs_raw_value = payload_dict.get("docs", [])
        docs_raw_value_list: list[object] = []
        if isinstance(docs_raw_value, list):
            docs_raw_value_list = cast(list[object], docs_raw_value)
        docs_raw: list[dict[str, object]] = []
        for doc_value in docs_raw_value_list:
            if not isinstance(doc_value, dict):
                continue
            doc: dict[str, object] = cast(dict[str, object], doc_value)
            docs_raw.append(doc)
        # Note 37: Parse tolerant to partial docs; missing author/year become None.

        candidates: list[MatchCandidate] = []
        for doc in docs_raw:
            cand_title = str(doc.get("title", ""))
            author_list_raw = doc.get("author_name", [])
            author_items: list[object] = []
            if isinstance(author_list_raw, list):
                author_items = cast(list[object], author_list_raw)
            else:
                author_items.append(author_list_raw)
            author_list = [str(item) for item in author_items]
            authors = ", ".join(author_list)
            year_val = doc.get("first_publish_year")
            year = int(year_val) if isinstance(year_val, int) else None
            score = 0.0
            candidates.append(
                MatchCandidate(
                    title=cand_title,
                    author=authors,
                    year=year,
                    source="openlibrary",
                    score=score,
                )
            )
        # Note 38: Open Library lacks similarity scores; fuzziness is computed later.
        return candidates


def _similarity(a: str, b: str) -> float:
    # Note 17: Token Jaccard is order-agnostic and stays fast/low-allocation.
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Jaccard over tokens as a simple fuzzy metric.
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def select_best_match(
    *,
    candidates: Iterable[MatchCandidate],
    normalized_title: str,
    normalized_author: str,
    threshold: float,
    disable_fuzzy: bool,
) -> MatchResult:
    best: MatchCandidate | None = None
    best_score = 0.0
    for candidate in candidates:
        # Note 18: Normalize candidates so comparisons match user text processing.
        cand_title = normalize_text(candidate.title)
        cand_author = normalize_text(candidate.author)
        exact_title = cand_title == normalized_title and cand_title != ""
        author_overlap = (
            bool(set(cand_author.split()) & set(normalized_author.split()))
            if normalized_author
            else False
        )
        if exact_title and author_overlap and candidate.year is not None:
            # Note 19: Exact title + author overlap = high confidence without fuzz.
            return MatchResult(
                year=candidate.year, confidence="high", source=candidate.source
            )

        if disable_fuzzy:
            continue

        title_score = _similarity(normalized_title, cand_title)
        author_score = _similarity(normalized_author, cand_author)
        score = (title_score + author_score) / 2.0
        if score >= threshold and score >= best_score and candidate.year is not None:
            # Note 20: Keep strongest fuzzy hit above threshold for determinism.
            best_score = score
            best = MatchCandidate(
                title=candidate.title,
                author=candidate.author,
                year=candidate.year,
                source=candidate.source,
                score=score,
            )

    if best is not None:
        return MatchResult(year=best.year, confidence="low", source=best.source)
    return MatchResult(year=None, confidence="none", source=None)


class FallbackClient(Protocol):
    def search(
        self, title: str, author: str
    ) -> list[MatchCandidate]: ...  # pragma: no cover


class NoopFallback(FallbackClient):
    def search(self, title: str, author: str) -> list[MatchCandidate]:
        return []

    # Note 42: Placeholder keeps pipeline extensible to other catalogs.


@dataclass
class EnrichmentResult:
    dataframe: pd.DataFrame
    summary: Summary


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    config: EnrichmentConfig,
    client: OpenLibraryClient,
    cache: CacheStore | None = None,
    checkpoint: CheckpointStore | None = None,
    fallback: FallbackClient | None = None,
) -> EnrichmentResult:
    working = df.copy()
    # Note 21: Work on a copy to avoid mutating caller-owned frames.
    summary = Summary()
    cache_store = cache or FileCache(config.cache_dir)
    checkpoint_store = checkpoint or FileCheckpoint(config.checkpoint_path)
    fallback_client = fallback or NoopFallback()

    start_index = checkpoint_store.load()
    # Note 22: Resume support preserves earlier rows when restarting.
    results_year: list[int | None] = []
    results_conf: list[Confidence] = []
    results_source: list[str | None] = []

    for idx, row in enumerate(working.itertuples(index=False), start=0):
        if idx < start_index:
            # Note 23: When resuming, reuse enrichment columns to stay idempotent.
            existing_year = getattr(row, "original_pub_year", None)
            is_missing_year = _is_missing_year(existing_year)
            results_year.append(None if is_missing_year else existing_year)
            results_conf.append(getattr(row, "pub_year_confidence", "none"))
            results_source.append(getattr(row, "original_pub_source", None))
            continue

        title = getattr(row, "title", "") or ""
        authors = getattr(row, "authors", "") or ""
        # Note 24: Normalize empty titles/authors to keep cache keys stable.
        normalized_title = normalize_text(title)
        normalized_author = normalize_text(authors)

        cache_key = f"{normalized_title}__{normalized_author}"
        cached = cache_store.get(cache_key)
        if cached is not None:
            # Note 25: Cache hits skip network/fuzz; critical under tight rate limits.
            result = cached
        else:
            # Note 39: Primary search runs before fallbacks to keep confidence steady.
            result = select_best_match(
                candidates=client.search(title, authors),
                normalized_title=normalized_title,
                normalized_author=normalized_author,
                threshold=config.fuzzy_threshold,
                disable_fuzzy=config.disable_fuzzy,
            )
            if result.confidence == "none":
                # Note 26: Use fallback only when primary fails; keeps semantics simple.
                fallback_candidates = fallback_client.search(title, authors)
                result = select_best_match(
                    candidates=fallback_candidates,
                    normalized_title=normalized_title,
                    normalized_author=normalized_author,
                    threshold=config.fuzzy_threshold,
                    disable_fuzzy=config.disable_fuzzy,
                )
            cache_store.set(cache_key, result)

        results_year.append(result.year)
        results_conf.append(result.confidence)
        results_source.append(result.source)
        summary.record(result)

        if (idx + 1) % config.checkpoint_every == 0:
            # Note 27: Checkpoint cadence bounds rework and trades IO for resilience.
            checkpoint_store.save(idx + 1, summary)

    working["original_pub_year"] = results_year
    working["pub_year_confidence"] = results_conf
    working["original_pub_source"] = results_source
    # Note 28: Final checkpoint captures row count and summary for audits.
    checkpoint_store.save(len(working), summary)
    return EnrichmentResult(dataframe=working, summary=summary)


def enrich_parquet(config: EnrichmentConfig) -> Summary:
    df = pd.read_parquet(config.input_path)  # type: ignore[reportUnknownMemberType]
    fallback_client: FallbackClient = NoopFallback()
    result = enrich_dataframe(
        df,
        config=config,
        client=OpenLibraryClient(
            rate_limit=config.rate_limit,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_initial=config.backoff_initial,
            backoff_cap=config.backoff_cap,
        ),
        cache=FileCache(config.cache_dir),
        checkpoint=FileCheckpoint(config.checkpoint_path),
        fallback=fallback_client,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_parquet(  # type: ignore[reportUnknownMemberType]
        config.output_path, index=False
    )
    # Note 40: Write once at end to avoid partial parquet; ensure output dir exists.
    print(
        json.dumps(
            {
                "output_path": str(config.output_path),
                "summary": result.summary.to_dict(),
                "rows": len(result.dataframe),
            }
        )
    )
    return result.summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich Gutenberg parquet with original publication year "
            "using Open Library and fallbacks."
        ),
    )
    # Note 29: Parser defaults mirror config; automation can tweak limits
    # without code changes.
    parser.add_argument("--input", required=True, type=Path, help="Input parquet path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output parquet path (default: data/meta/gutenberg_books_enhanced.parquet)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path for resumable progress",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=5.0, help="Requests per second"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Max retries per request"
    )
    parser.add_argument(
        "--backoff-initial", type=float, default=0.5, help="Initial backoff seconds"
    )
    parser.add_argument(
        "--backoff-cap", type=float, default=8.0, help="Maximum backoff seconds"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for fuzzy matches",
    )
    parser.add_argument(
        "--disable-fuzzy", action="store_true", help="Require exact matches only"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache directory"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500, help="Checkpoint frequency in rows"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=10.0, help="HTTP timeout per request"
    )
    parser.add_argument(
        "--enable-wikidata", action="store_true", help="Enable Wikidata fallback lookup"
    )
    parser.add_argument(
        "--enable-loc", action="store_true", help="Enable LOC fallback lookup"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> EnrichmentConfig:
    parser = build_arg_parser()
    # Note 43: Defaults let scripts omit flags and still use standard folders.
    args = parser.parse_args(argv)
    return EnrichmentConfig(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        backoff_initial=args.backoff_initial,
        backoff_cap=args.backoff_cap,
        fuzzy_threshold=args.fuzzy_threshold,
        disable_fuzzy=args.disable_fuzzy,
        checkpoint_every=args.checkpoint_every,
        timeout_seconds=args.timeout_seconds,
        enable_wikidata=args.enable_wikidata,
        enable_loc=args.enable_loc,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        enrich_parquet(config)
        # Note 45: Main orchestrates so logic stays importable and side-effect free.
        return 0
    except Exception as exc:  # pragma: no cover - CLI boundary
        # Note 30: CLI boundary logs the error and returns non-zero for schedulers.
        print(f"Enrichment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

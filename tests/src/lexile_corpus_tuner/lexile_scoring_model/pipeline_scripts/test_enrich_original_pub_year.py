from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pandas as pd
import pytest  # noqa: TCH002 - pytest is required at runtime for tests
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)
from requests import Response

EnrichmentConfig = enrich.EnrichmentConfig
MatchCandidate = enrich.MatchCandidate
MatchResult = enrich.MatchResult
OpenLibraryClient = enrich.OpenLibraryClient
Summary = enrich.Summary
enrich_dataframe = enrich.enrich_dataframe
normalize_text = enrich.normalize_text
parse_args = enrich.parse_args
select_best_match = enrich.select_best_match


class MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, MatchResult] = {}

    def get(self, key: str) -> MatchResult | None:
        return self.store.get(key)

    def set(self, key: str, value: MatchResult) -> None:
        self.store[key] = value


class MemoryCheckpoint:
    def __init__(self, start: int = 0) -> None:
        self.start = start
        self.saved: tuple[int, Summary] | None = None

    def load(self) -> int:
        return self.start

    def save(self, index: int, summary: Summary) -> None:
        self.saved = (index, summary)


class FakeClient:
    def __init__(self, candidates: list[MatchCandidate] | None = None) -> None:
        self._candidates = candidates or []
        self.calls = 0

    def search(self, title: str, author: str) -> list[MatchCandidate]:
        self.calls += 1
        return list(self._candidates)


class FakeFallback(FakeClient):
    pass


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 1, "title": "Sample Book", "authors": "Jane Doe"},
            {"id": 2, "title": "Another Tale", "authors": "John Smith"},
        ]
    )


def test_normalize_text_strips_punctuation_and_casefolds() -> None:
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("  Spaced\tText ") == "spaced text"


def test_select_best_match_exact_high() -> None:
    candidates = [
        MatchCandidate(
            title="Sample Book",
            author="Jane Doe",
            year=1999,
            source="openlibrary",
            score=0.0,
        )
    ]
    result = select_best_match(
        candidates=candidates,
        normalized_title="sample book",
        normalized_author="jane doe",
        threshold=0.9,
        disable_fuzzy=False,
    )
    assert result.confidence == "high"
    assert result.year == 1999
    assert result.source == "openlibrary"


def test_select_best_match_fuzzy_low_and_none() -> None:
    candidates = [
        MatchCandidate(
            title="Sample Bk",
            author="J. Doe",
            year=2001,
            source="openlibrary",
            score=0.0,
        ),
    ]
    low = select_best_match(
        candidates=candidates,
        normalized_title="sample book",
        normalized_author="jane doe",
        threshold=0.2,
        disable_fuzzy=False,
    )
    assert low.confidence == "low"
    none_result = select_best_match(
        candidates=candidates,
        normalized_title="unrelated title",
        normalized_author="someone else",
        threshold=0.9,
        disable_fuzzy=False,
    )
    assert none_result.confidence == "none"


def test_select_best_match_missing_year_returns_none() -> None:
    candidates = [
        MatchCandidate(
            title="Sample", author="Jane", year=None, source="openlibrary", score=0.0
        )
    ]
    result = select_best_match(
        candidates=candidates,
        normalized_title="sample",
        normalized_author="jane",
        threshold=0.1,
        disable_fuzzy=False,
    )
    assert result.confidence == "none"
    assert result.year is None


def test_enrich_dataframe_uses_cache_and_skips_client_call() -> None:
    df = make_df().head(1)
    cache = MemoryCache()
    cache.set(
        "sample book__jane doe",
        MatchResult(year=1980, confidence="high", source="cached"),
    )
    client = FakeClient([])
    result = enrich_dataframe(
        df,
        config=EnrichmentConfig(input_path=Path("in")),
        client=client,  # type: ignore[arg-type]
        cache=cache,
        checkpoint=MemoryCheckpoint(),
    )
    assert client.calls == 0
    assert result.dataframe.loc[0, "original_pub_year"] == 1980
    assert result.dataframe.loc[0, "pub_year_confidence"] == "high"
    assert result.dataframe.loc[0, "original_pub_source"] == "cached"


def test_enrich_dataframe_uses_fallback_when_primary_none() -> None:
    df = make_df().head(1)
    primary = FakeClient([])
    fallback = FakeFallback(
        [
            MatchCandidate(
                title="Sample Book",
                author="Jane Doe",
                year=1950,
                source="wikidata",
                score=0.0,
            )
        ]
    )
    result = enrich_dataframe(
        df,
        config=EnrichmentConfig(input_path=Path("in")),
        client=primary,  # type: ignore[arg-type]
        cache=MemoryCache(),
        checkpoint=MemoryCheckpoint(),
        fallback=fallback,  # type: ignore[arg-type]
    )
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.dataframe.loc[0, "original_pub_year"] == 1950
    assert result.dataframe.loc[0, "original_pub_source"] == "wikidata"


def test_enrich_dataframe_checkpoint_resume_skips_completed_rows() -> None:
    df = make_df()
    client = FakeClient(
        [
            MatchCandidate(
                title="Another Tale",
                author="John Smith",
                year=2005,
                source="openlibrary",
                score=0.0,
            )
        ]
    )
    checkpoint = MemoryCheckpoint(start=1)
    result = enrich_dataframe(
        df,
        config=EnrichmentConfig(input_path=Path("in")),
        client=client,  # type: ignore[arg-type]
        cache=MemoryCache(),
        checkpoint=checkpoint,
    )
    value = cast(object, result.dataframe.loc[0, "original_pub_year"])
    # Pyright pandas stubs mark isna partially unknown for object inputs.
    assert pd.isna(value)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert result.dataframe.loc[1, "original_pub_year"] == 2005
    assert checkpoint.saved is not None


def test_enrich_dataframe_summarizes_matches() -> None:
    df = make_df()
    client = FakeClient(
        [
            MatchCandidate(
                title="Sample Book",
                author="Jane Doe",
                year=1990,
                source="openlibrary",
                score=0.0,
            ),
            MatchCandidate(
                title="Another Tale",
                author="John Smith",
                year=2001,
                source="openlibrary",
                score=0.0,
            ),
        ]
    )
    summary = enrich_dataframe(
        df,
        config=EnrichmentConfig(input_path=Path("in"), disable_fuzzy=True),
        client=client,  # type: ignore[arg-type]
        cache=MemoryCache(),
        checkpoint=MemoryCheckpoint(),
    ).summary
    assert summary.matched_high == 2
    assert summary.matched_low == 0
    assert summary.matched_none == 0


def test_parse_args_sets_defaults_and_overrides() -> None:
    args = parse_args(
        [
            "--input",
            "input.parquet",
            "--output",
            "out.parquet",
            "--rate-limit",
            "10",
            "--fuzzy-threshold",
            "0.5",
            "--disable-fuzzy",
        ]
    )
    assert args.input_path.name == "input.parquet"
    assert args.output_path.name == "out.parquet"
    assert args.rate_limit == 10
    assert args.fuzzy_threshold == 0.5
    assert args.disable_fuzzy is True


def test_openlibrary_client_respects_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # minimal smoke: ensure search calls HTTP once and returns candidates
    class StubResponse(Response):
        def __init__(self, payload: dict[str, object]):
            super().__init__()
            self._payload = payload

        def raise_for_status(self) -> None:  # type: ignore[override]
            return None

        def json(self, **_kwargs: object) -> dict[str, object]:  # type: ignore[override]
            return self._payload

    calls: list[float] = []

    class StubHttp:
        def get(  # type: ignore[override]
            self, url: str, *, params: dict[str, str], timeout: float
        ) -> Response:
            calls.append(time.time())
            return StubResponse(
                {
                    "docs": [
                        {
                            "title": "Book",
                            "author_name": ["A"],
                            "first_publish_year": 1900,
                        }
                    ]
                }
            )

    client = OpenLibraryClient(
        http=cast(enrich.HttpClient, StubHttp()), rate_limit=1000.0, timeout_seconds=1.0
    )
    result = client.search("Book", "A")
    assert result[0].year == 1900
    assert len(calls) == 1

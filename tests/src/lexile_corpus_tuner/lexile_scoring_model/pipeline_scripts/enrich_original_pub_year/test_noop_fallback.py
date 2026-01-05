from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)

NoopFallback = enrich.NoopFallback


def test_noop_fallback_returns_fresh_empty_list() -> None:
    fallback = NoopFallback()

    first = fallback.search("title", "author")
    second = fallback.search("title", "author")

    assert first == []
    assert second == []
    assert first is not second

from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)

MatchResult = enrich.MatchResult
Summary = enrich.Summary


def test_summary_records_counts_and_errors() -> None:
    summary = Summary()

    summary.record(MatchResult(year=2001, confidence="high", source="s"))
    summary.record(MatchResult(year=2002, confidence="low", source="s"))
    summary.record(MatchResult(year=2003, confidence="none", source="s"))
    summary.record_error()

    assert summary.matched_high == 1
    assert summary.matched_low == 1
    assert summary.matched_none == 1
    assert summary.errors == 1
    assert summary.to_dict() == {
        "matched_high": 1,
        "matched_low": 1,
        "matched_none": 1,
        "errors": 1,
    }


def test_summary_treats_unknown_confidence_as_none() -> None:
    summary = Summary()

    summary.record(MatchResult(year=None, confidence="none", source="s"))
    summary.record(MatchResult(year=1990, confidence="none", source="s"))

    assert summary.matched_high == 0
    assert summary.matched_low == 0
    assert summary.matched_none == 2

from __future__ import annotations

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as pub_year,
)


def test_select_best_match_prefers_earliest_exact_year_for_duplicates() -> None:
    """
    Exact title/author duplicates with differing years should return the earliest year.

    This regression guards against the current order bias where the first exact
    match short-circuits selection, even when an earlier publication year exists
    later in the candidate list.
    """

    normalized_title = pub_year.normalize_text("Moby-Dick; or, the Whale")
    normalized_author = pub_year.normalize_text("Herman Melville")

    candidates = [
        pub_year.MatchCandidate(
            title="Moby Dick; or, the Whale",
            author="Herman Melville",
            year=2014,
            source="openlibrary",
            score=0.0,
        ),
        pub_year.MatchCandidate(
            title="Moby-Dick or, the Whale",
            author="Herman Melville",
            year=2020,
            source="openlibrary",
            score=0.0,
        ),
        pub_year.MatchCandidate(
            title="Moby Dick; or, the Whale",
            author="Herman Melville",
            year=2017,
            source="openlibrary",
            score=0.0,
        ),
        pub_year.MatchCandidate(
            title="Moby Dick or The Whale",
            author="Herman Melville",
            year=1956,
            source="openlibrary",
            score=0.0,
        ),
        pub_year.MatchCandidate(
            title="Moby-Dick; or, the WHALE",
            author="Herman Melville",
            year=2019,
            source="openlibrary",
            score=0.0,
        ),
    ]

    result = pub_year.select_best_match(
        candidates=candidates,
        normalized_title=normalized_title,
        normalized_author=normalized_author,
        threshold=0.9,
        disable_fuzzy=False,
    )

    assert result.year == 1956
    assert result.confidence == "high"
    assert result.source == "openlibrary"

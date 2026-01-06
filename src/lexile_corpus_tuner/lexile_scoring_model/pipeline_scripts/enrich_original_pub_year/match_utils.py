"""
Text normalization and matching utilities for publication-year enrichment.

Purpose:
    Provide deterministic normalization, similarity scoring, and candidate
    selection helpers used by the enrichment pipeline.

Side Effects:
    None; all helpers are pure.
"""

from __future__ import annotations

import math
import numbers
import re
from typing import TYPE_CHECKING

from .match_candidate import MatchCandidate
from .match_result import MatchResult

if TYPE_CHECKING:
    from collections.abc import Iterable


def normalize_text(value: str) -> str:
    """
    Normalize text for stable matching by removing punctuation and collapsing space.

    Purpose:
        Provide deterministic normalization so cache keys and similarity checks are
        consistent across runs.

    Args:
        value (str): Raw title or author value from the dataset.

    Returns:
        str: Lowercased, punctuation-stripped, whitespace-collapsed string.

    Side Effects:
        None.
    """

    cleaned = re.sub(r"[^\w\s]", " ", value).lower()
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def is_missing_year(value: object) -> bool:
    """
    Detect missing year values, treating NaN-like values as absent but keeping zeros.

    Purpose:
        Distinguish truly missing data from valid numeric values so enrichment does
        not overwrite legitimate zeros.

    Args:
        value (object): Original value from the dataframe.

    Returns:
        bool: True when the value should be considered missing.

    Side Effects:
        None.
    """

    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return math.isnan(float(value))
    return False


def _similarity(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity between two normalized strings.

    Purpose:
        Score overlap between candidate and target strings for fuzzy matching.

    Args:
        a (str): Normalized string to compare.
        b (str): Normalized string to compare.

    Returns:
        float: Jaccard similarity in the range [0.0, 1.0].

    Side Effects:
        None.
    """

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
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
    """
    Choose the strongest candidate, preferring exact matches then fuzzy scoring.

    Purpose:
        Encapsulate match selection rules so enrichment can remain deterministic and
        testable.

    Args:
        candidates (Iterable[MatchCandidate]): Provider results to evaluate.
        normalized_title (str): Normalized target title.
        normalized_author (str): Normalized target author list.
        threshold (float): Minimum similarity required for fuzzy acceptance.
        disable_fuzzy (bool): When True, only exact matches are eligible.

    Returns:
        MatchResult: Selected year with confidence and source metadata.

    Side Effects:
        None.
    """

    best: MatchCandidate | None = None
    best_score = 0.0
    earliest_exact_year: int | None = None
    earliest_exact_source: str | None = None
    # Track earliest exact match; fall back to fuzzy scoring when none exist.
    for candidate in candidates:
        cand_title = normalize_text(candidate.title)
        cand_author = normalize_text(candidate.author)
        exact_title = cand_title == normalized_title and cand_title != ""
        author_overlap = (
            bool(set(cand_author.split()) & set(normalized_author.split()))
            if normalized_author
            else False
        )
        # Accumulate the earliest exact title match with overlapping authors.
        if exact_title and author_overlap and candidate.year is not None:
            if earliest_exact_year is None or candidate.year < earliest_exact_year:
                earliest_exact_year = candidate.year
                earliest_exact_source = candidate.source
            continue

        if disable_fuzzy:
            continue

        title_score = _similarity(normalized_title, cand_title)
        author_score = _similarity(normalized_author, cand_author)
        score = (title_score + author_score) / 2.0
        # Track the best-scoring fuzzy candidate above the acceptance threshold.
        if score >= threshold and score >= best_score and candidate.year is not None:
            best_score = score
            best = MatchCandidate(
                title=candidate.title,
                author=candidate.author,
                year=candidate.year,
                source=candidate.source,
                score=score,
            )

    if earliest_exact_year is not None and earliest_exact_source is not None:
        return MatchResult(
            year=earliest_exact_year, confidence="high", source=earliest_exact_source
        )

    if best is not None:
        return MatchResult(year=best.year, confidence="low", source=best.source)
    return MatchResult(year=None, confidence="none", source=None)

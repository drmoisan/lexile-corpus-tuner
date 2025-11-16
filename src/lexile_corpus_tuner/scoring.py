from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .estimators import LexileEstimator
from .models import DocumentLexileStats, Window, WindowScore


def score_windows(windows: List[Window], estimator: LexileEstimator) -> List[WindowScore]:
    """Score each window using the provided estimator."""
    scores: List[WindowScore] = []
    for window in windows:
        lexile = estimator.predict_scalar(window.text)
        scores.append(WindowScore(window=window, lexile=lexile))
    return scores


def smooth_window_lexiles(window_scores: List[WindowScore], kernel_size: int) -> List[float]:
    """
    Apply a moving average smoothing across window Lexile values.
    kernel_size <= 1 simply returns the raw lexiles.
    """
    if kernel_size <= 1 or not window_scores:
        return [score.lexile for score in window_scores]

    values = [score.lexile for score in window_scores]
    length = len(values)
    smoothed: List[float] = []
    half = kernel_size // 2

    for idx in range(length):
        start = max(0, idx - half)
        end = min(length, idx + half + 1)
        window_vals = values[start:end]
        smoothed.append(sum(window_vals) / len(window_vals))

    return smoothed


def compute_document_stats(all_window_scores: List[WindowScore]) -> List[DocumentLexileStats]:
    """Group window scores by document and compute aggregate Lexile statistics."""
    grouped: Dict[str, List[WindowScore]] = defaultdict(list)
    for score in all_window_scores:
        grouped[score.window.doc_id].append(score)

    stats: List[DocumentLexileStats] = []
    for doc_id, scores in grouped.items():
        avg = sum(s.lexile for s in scores) / len(scores) if scores else 0.0
        max_score = max((s.lexile for s in scores), default=0.0)
        stats.append(
            DocumentLexileStats(
                doc_id=doc_id,
                avg_lexile=avg,
                max_lexile=max_score,
                window_scores=list(scores),
            )
        )

    return stats

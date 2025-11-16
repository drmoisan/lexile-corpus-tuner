from lexile_corpus_tuner.config import LexileTunerConfig
from lexile_corpus_tuner.constraints import find_violations, has_hard_window_violations
from lexile_corpus_tuner.estimators.base import LexileEstimator
from lexile_corpus_tuner.models import DocumentLexileStats, Window, WindowScore
from lexile_corpus_tuner.scoring import (
    compute_document_stats,
    score_windows,
    smooth_window_lexiles,
)


class ConstantEstimator(LexileEstimator):
    def __init__(self, value: float) -> None:
        self.value = value

    def predict_scalar(self, text: str) -> float:  # pragma: no cover - trivial
        return float(self.value)


def _make_windows() -> list[Window]:
    return [
        Window("doc", 0, 0, 1, "alpha"),
        Window("doc", 1, 1, 2, "beta"),
        Window("doc", 2, 2, 3, "gamma"),
    ]


def test_scoring_and_stats_pipeline():
    windows = _make_windows()
    estimator = ConstantEstimator(300)
    scored = score_windows(windows, estimator)
    assert all(score.lexile == 300 for score in scored)

    smoothed = smooth_window_lexiles(
        [
            WindowScore(window=windows[0], lexile=100),
            WindowScore(window=windows[1], lexile=200),
            WindowScore(window=windows[2], lexile=300),
        ],
        kernel_size=3,
    )
    assert smoothed == [150.0, 200.0, 250.0]

    stats = compute_document_stats(scored)
    assert stats[0].avg_lexile == 300
    assert stats[0].max_lexile == 300


def test_find_violations_detects_window_and_avg():
    windows = _make_windows()
    window_scores = [
        WindowScore(window=windows[0], lexile=250),
        WindowScore(window=windows[1], lexile=470),
    ]
    stats = DocumentLexileStats(
        doc_id="doc",
        avg_lexile=400,
        max_lexile=470,
        window_scores=window_scores,
    )

    config = LexileTunerConfig(
        max_window_lexile=300,
        target_avg_lexile=200,
        avg_tolerance=50,
    )
    violations = find_violations(stats, config)
    assert has_hard_window_violations(violations)
    reasons = [v.reason for v in violations]
    assert any("Window Lexile" in reason for reason in reasons)
    assert any("Document average" in reason for reason in reasons)

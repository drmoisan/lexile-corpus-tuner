from lexile_corpus_tuner.config import LexileTunerConfig
from lexile_corpus_tuner.estimators.base import LexileEstimator
from lexile_corpus_tuner.models import Document
from lexile_corpus_tuner.pipeline import process_document
from lexile_corpus_tuner.rewriting import NoOpRewriter, Rewriter


class ThresholdEstimator(LexileEstimator):
    def __init__(self, trigger: str, low: float = 300.0, high: float = 600.0) -> None:
        self.trigger = trigger
        self.low = low
        self.high = high

    def predict_scalar(self, text: str) -> float:  # pragma: no cover - trivial
        return self.high if self.trigger in text else self.low


class SimpleRewriter(Rewriter):
    def rewrite(self, text: str, target_lexile: float) -> str:  # pragma: no cover - trivial
        return text.replace("complex", "simple")


def test_process_document_without_rewrite():
    """Pipeline leaves document unchanged when rewriting is disabled."""
    doc = Document(doc_id="doc", text="Short friendly sentences keep things easy.")
    config = LexileTunerConfig(
        window_size=50,
        stride=25,
        max_window_lexile=900,
        rewrite_enabled=False,
        target_avg_lexile=300,
        avg_tolerance=100,
    )
    estimator = ThresholdEstimator(trigger="impossible")
    final_doc, stats, violations = process_document(doc, config, estimator, NoOpRewriter())

    assert final_doc.text == doc.text
    assert stats.max_lexile == stats.avg_lexile == 300
    assert violations == []


def test_process_document_with_simple_rewrite():
    """Simple rewriter reduces lexile violations by replacing terms."""
    doc = Document(
        doc_id="doc",
        text="This is a complex sentence that should become simpler after rewriting.",
    )
    config = LexileTunerConfig(
        window_size=200,
        stride=50,
        max_window_lexile=450,
        rewrite_enabled=True,
        max_passes=2,
        target_avg_lexile=350,
        avg_tolerance=200,
    )
    estimator = ThresholdEstimator(trigger="complex", high=600, low=300)
    final_doc, stats, violations = process_document(doc, config, estimator, SimpleRewriter())

    assert "complex" not in final_doc.text
    assert stats.max_lexile <= config.max_window_lexile
    assert not any(v.window_id >= 0 for v in violations)

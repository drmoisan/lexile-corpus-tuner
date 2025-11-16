from __future__ import annotations

import re
from statistics import mean

from .base import LexileEstimator

WORD_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


class DummyLexileEstimator(LexileEstimator):
    """
    Simple heuristic estimator that correlates sentence and word length
    with Lexile difficulty. This keeps the pipeline runnable without external
    ML dependencies.
    """

    def predict_scalar(self, text: str) -> float:
        words = WORD_RE.findall(text)
        if not words:
            return 0.0

        sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        avg_chars_per_word = mean(len(word) for word in words)
        avg_words_per_sentence = (
            len(words) / len(sentences) if sentences else float(len(words))
        )

        # Linear heuristic scaled to roughly match Lexile magnitudes.
        lexile = 200.0 + 12.0 * avg_words_per_sentence + 25.0 * avg_chars_per_word
        return float(lexile)

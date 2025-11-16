from __future__ import annotations

from abc import ABC, abstractmethod


class LexileEstimator(ABC):
    """Abstract estimator that provides Lexile-like scalar predictions."""

    @abstractmethod
    def predict_scalar(self, text: str) -> float:
        """Return a numeric Lexile-like difficulty score for the input text."""
        raise NotImplementedError

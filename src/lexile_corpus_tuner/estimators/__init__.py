from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import LexileEstimator
from .dummy_estimator import DummyLexileEstimator
from .lexile_determination_v2_adapter import LexileDeterminationV2Estimator

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ..config import LexileTunerConfig

__all__ = [
    "LexileEstimator",
    "DummyLexileEstimator",
    "LexileDeterminationV2Estimator",
    "create_estimator",
    "build_estimator_from_config",
]


def create_estimator(name: str, **kwargs: Any) -> LexileEstimator:
    """Factory for building estimators by name."""
    normalized = name.lower().strip()
    if normalized == "dummy":
        return DummyLexileEstimator()
    if normalized in {"lexile_v2", "lexile_determination_v2"}:
        return LexileDeterminationV2Estimator(**kwargs)
    raise ValueError(f"Unknown estimator '{name}'.")


def build_estimator_from_config(config: "LexileTunerConfig") -> LexileEstimator:
    """Convenience helper to build an estimator from LexileTunerConfig."""
    normalized = config.estimator_name.lower().strip()
    if normalized in {"lexile_v2", "lexile_determination_v2"}:
        return create_estimator(
            config.estimator_name,
            model_path=config.lexile_v2_model_path or "",
            vectorizer_path=config.lexile_v2_vectorizer_path or "",
            label_encoder_path=config.lexile_v2_label_encoder_path,
            stopwords_path=config.lexile_v2_stopwords_path,
            band_to_midpoint=config.lexile_v2_band_to_midpoint,
        )
    return create_estimator(config.estimator_name)

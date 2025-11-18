from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..calibration.featureset import make_regression_features
from ..calibration.model_store import load_model_spec
from .features import DocumentFeatures

MODEL_PATH = Path("data/model/lexile_regression_model.json")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Regression model spec not found at {MODEL_PATH}. "
            "Run calibration fit to create it."
        )
    return load_model_spec(MODEL_PATH)


def estimate_lexile_from_features(features: DocumentFeatures) -> float:
    """Estimate a Lexile-like measure from features using stored coefficients."""
    spec = _load_model()
    feat_dict = make_regression_features(features)

    coefficients = spec.get("coefficients", [])
    feature_names = spec.get("features", [])
    intercept = float(spec.get("intercept", 0.0))

    if len(coefficients) != len(feature_names):
        raise ValueError("Model spec coefficients/features mismatch.")

    y_hat = intercept
    for coef, feature_name in zip(coefficients, feature_names):
        value = float(feat_dict.get(feature_name, 0.0))
        y_hat += float(coef) * value
    return float(y_hat)

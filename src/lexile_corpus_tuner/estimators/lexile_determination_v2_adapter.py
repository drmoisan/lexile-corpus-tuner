from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .base import LexileEstimator

load_model: Any | None = None
joblib: Any | None = None


class LexileDeterminationV2Estimator(LexileEstimator):
    """
    Adapter around the external lexile-determination-v2 Keras classifier.

    Responsibilities:
    * Load a trained model + vectorizer (and optional label encoder) exported by the
      lexile-determination-v2 project.
    * Predict a Lexile band/label for the provided text.
    * Convert that label into a scalar Lexile value for use in this pipeline.

    TODO: Align artifact loading with the authoritative lexile-determination-v2 repo.
    """

    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        label_encoder_path: Optional[str] = None,
        band_to_midpoint: Optional[Mapping[str, float]] = None,
    ) -> None:
        tf_load_model = _ensure_tensorflow()
        joblib_module = _ensure_joblib()

        if not model_path or not vectorizer_path:
            raise ValueError(
                "model_path and vectorizer_path must be provided for LexileDeterminationV2Estimator."
            )

        # TODO: Ensure the load call matches how lexile-determination-v2 persists its model.
        self.model = tf_load_model(model_path)
        self.vectorizer = joblib_module.load(vectorizer_path)
        self.label_encoder = (
            joblib_module.load(label_encoder_path) if label_encoder_path else None
        )
        self._band_to_midpoint: Dict[str, float] = dict(band_to_midpoint or {})
        # TODO: Populate default mapping when lexile-determination-v2 exposes class indices.
        self._index_to_band: Dict[int, str] = {}

    def predict_scalar(self, text: str) -> float:
        """
        Predict a numeric Lexile approximation for the provided text by:
        1. Applying the saved vectorizer.
        2. Running the classifier.
        3. Mapping the predicted class label to a scalar.
        """
        vector = self.vectorizer.transform([text])
        model_input = vector.toarray() if hasattr(vector, "toarray") else vector
        probabilities = self.model.predict(model_input, verbose=0)

        probs = probabilities[0] if probabilities.ndim > 1 else probabilities
        class_idx = int(np.argmax(probs))
        label = self._index_to_label(class_idx)
        return float(self._label_to_numeric_lexile(label))

    def _index_to_label(self, idx: int) -> str:
        """
        Map a class index to a band label. Requires either a label encoder or a
        fallback index -> band mapping.
        """
        if self.label_encoder is not None:
            decoded = self.label_encoder.inverse_transform([idx])
            return str(decoded[0])

        if idx in self._index_to_band:
            return self._index_to_band[idx]

        raise RuntimeError(
            "No label encoder provided and _index_to_band is empty. "
            "Provide a label encoder artifact or hard-code index mappings."
        )

    def _label_to_numeric_lexile(self, label: str) -> float:
        """
        Convert human-readable Lexile band labels into numeric approximations.

        Supports:
        * '900-999' or '900 - 999L'
        * '950L'
        * Explicit mappings defined in config.
        """
        label = label.strip()
        if label in self._band_to_midpoint:
            return self._band_to_midpoint[label]

        range_match = re.match(r"^\s*(\d+)\s*-\s*(\d+)", label)
        if range_match:
            low = int(range_match.group(1))
            high = int(range_match.group(2))
            return (low + high) / 2.0

        numeric_match = re.search(r"(\d+)", label)
        if numeric_match:
            return float(numeric_match.group(1))

        raise ValueError(f"Cannot parse Lexile label {label!r}")


def _ensure_tensorflow() -> Any:
    global load_model
    if load_model is not None:
        return load_model
    try:  # pragma: no cover - import guard
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "TensorFlow/Keras is required for LexileDeterminationV2Estimator. "
            "Install the 'lexile-v2' extra via `pip install .[lexile-v2]`."
        ) from exc
    load_model = keras_load_model
    return load_model


def _ensure_joblib() -> Any:
    global joblib
    if joblib is not None:
        return joblib
    try:  # pragma: no cover - import guard
        import joblib as joblib_module  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "joblib is required for LexileDeterminationV2Estimator. "
            "Install the 'lexile-v2' extra via `pip install .[lexile-v2]`."
        ) from exc
    joblib = joblib_module
    return joblib

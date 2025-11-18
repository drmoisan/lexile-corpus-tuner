from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Mapping, cast

import numpy as np

from .base import LexileEstimator
from .lexile_v2_preprocessing import load_stopwords as load_lexile_stopwords
from .lexile_v2_preprocessing import (
    vectorize_with_lexile_pipeline,
)

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
        label_encoder_path: str | None = None,
        stopwords_path: str | None = None,
        band_to_midpoint: Mapping[str, float] | None = None,
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
        label_payload = (
            joblib_module.load(label_encoder_path) if label_encoder_path else None
        )
        self._label_mapping: list[str] | None = None
        if label_payload is not None and hasattr(label_payload, "inverse_transform"):
            self.label_encoder = label_payload
        elif label_payload is not None:
            self.label_encoder = None
            self._label_mapping = [str(value) for value in list(label_payload)]
        else:
            self.label_encoder = None
        self._stopwords = (
            load_lexile_stopwords(stopwords_path)
            if stopwords_path is not None
            else None
        )
        self._use_lexile_pipeline = bool(
            self._stopwords and hasattr(self.vectorizer, "texts_to_matrix")
        )
        self._band_to_midpoint: dict[str, float] = dict(band_to_midpoint or {})
        # TODO: Populate default mapping when lexile-determination-v2 exposes class indices.
        self._index_to_band: dict[int, str] = {}

    def predict_scalar(self, text: str) -> float:
        """
        Predict a numeric Lexile approximation for the provided text by:
        1. Applying the saved vectorizer.
        2. Running the classifier.
        3. Mapping the predicted class label to a scalar.
        """
        model_input = self._preprocess_text(text)
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
        if self._label_mapping is not None and 0 <= idx < len(self._label_mapping):
            return self._label_mapping[idx]

        if idx in self._index_to_band:
            return self._index_to_band[idx]

        raise RuntimeError(
            "No label encoder provided and _index_to_band is empty. "
            "Provide a label encoder artifact or hard-code index mappings."
        )

    def _preprocess_text(self, text: str) -> Any:
        if self._use_lexile_pipeline:
            matrix = vectorize_with_lexile_pipeline(
                text, self.vectorizer, self._stopwords or []
            )
            array = np.asarray(matrix)
            if array.ndim == 2:
                array = np.expand_dims(array, axis=1)
            return array
        vector = self.vectorizer.transform([text])
        return vector.toarray() if hasattr(vector, "toarray") else vector

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


def _ensure_tensorflow() -> Callable[..., Any]:
    global load_model
    if load_model is not None:
        return load_model

    try:  # pragma: no cover - import guard
        keras_models = cast(Any, importlib.import_module("tensorflow.keras.models"))
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "TensorFlow/Keras is required for LexileDeterminationV2Estimator. "
            "Install the 'lexile-v2' extra via `pip install .[lexile-v2]`."
        ) from exc
    keras_load_model = getattr(keras_models, "load_model", None)
    if not callable(keras_load_model):  # pragma: no cover - defensive
        raise ImportError(
            "tensorflow.keras.models.load_model is unavailable in the current installation."
        )
    load_model = cast(Callable[..., Any], keras_load_model)
    return load_model


def _ensure_joblib() -> Any:
    global joblib
    if joblib is not None:
        return joblib
    try:  # pragma: no cover - import guard
        joblib_module = cast(Any, importlib.import_module("joblib"))
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "joblib is required for LexileDeterminationV2Estimator. "
            "Install the 'lexile-v2' extra via `pip install .[lexile-v2]`."
        ) from exc
    joblib = joblib_module
    return joblib

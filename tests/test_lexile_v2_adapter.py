import types

import numpy as np

import lexile_corpus_tuner.estimators.lexile_determination_v2_adapter as adapter


class DummyModel:
    def predict(self, X, verbose: int = 0):
        return np.array([[0.1, 0.8, 0.1]])


class DummyVectorizer:
    def transform(self, texts):
        return np.array([[1.0, 2.0, 3.0]])


class DummyLabelEncoder:
    def inverse_transform(self, indices):
        return np.array(["900-999"])


def test_predict_scalar_with_stubbed_artifacts(monkeypatch):
    """Stubbed model artifacts yield the expected midpoint lexile score."""
    monkeypatch.setattr(adapter, "load_model", lambda path: DummyModel())
    monkeypatch.setattr(
        adapter,
        "joblib",
        types.SimpleNamespace(
            load=lambda path: DummyVectorizer()
            if "vectorizer" in path
            else DummyLabelEncoder()
        ),
    )

    estimator = adapter.LexileDeterminationV2Estimator(
        model_path="dummy_model.h5",
        vectorizer_path="dummy_vectorizer.pkl",
        label_encoder_path="dummy_labels.pkl",
    )

    score = estimator.predict_scalar("Some input text.")
    assert abs(score - 949.5) < 1e-6


def test_predict_scalar_with_index_mapping(monkeypatch):
    """Estimator falls back to index-to-band mapping when labels missing."""
    monkeypatch.setattr(adapter, "load_model", lambda path: DummyModel())
    monkeypatch.setattr(
        adapter,
        "joblib",
        types.SimpleNamespace(
            load=lambda path: DummyVectorizer()
        ),
    )

    estimator = adapter.LexileDeterminationV2Estimator(
        model_path="dummy_model.h5",
        vectorizer_path="dummy_vectorizer.pkl",
        band_to_midpoint={"700-799": 750.0},
    )
    estimator._index_to_band[1] = "700-799"  # type: ignore[attr-defined]
    score = estimator.predict_scalar("Another snippet.")
    assert score == 750.0

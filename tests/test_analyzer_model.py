# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
"""Tests for analyzer model module.

This module tests the model loading and Lexile estimation functions
that use stored regression model coefficients.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from lexile_corpus_tuner.lexile_scoring_model.analyzer.features import (
    DocumentFeatures,
    SliceFeatures,
)
from lexile_corpus_tuner.lexile_scoring_model.analyzer.model import (
    MODEL_PATH,
    _load_model,
    estimate_lexile_from_features,
)

ANALYZER_MODEL_PATH = "lexile_corpus_tuner.lexile_scoring_model.analyzer.model"


@pytest.fixture
def sample_model_spec() -> dict[str, Any]:
    """Create a sample model specification for testing."""
    return {
        "version": "2024-01-01",
        "features": [
            "overall_msl",
            "overall_mlf",
            "log_num_tokens",
            "msl_std",
            "mlf_std",
            "overall_msl_sq",
            "overall_mlf_sq",
            "msl_times_mlf",
        ],
        "coefficients": [10.0, -50.0, 20.0, 5.0, -5.0, 0.5, 1.0, 2.0],
        "intercept": 500.0,
        "metrics": {"rmse": 50.0, "mae": 40.0, "r": 0.85},
    }


@pytest.fixture
def sample_document_features() -> DocumentFeatures:
    """Create sample document features for testing."""
    slice_feat = SliceFeatures(
        slice_id=0,
        num_tokens=100,
        num_sentences=10,
        mean_sentence_length=10.0,
        mean_log_word_freq=3.0,
    )
    return DocumentFeatures(
        num_slices=1,
        total_tokens=100,
        overall_mean_sentence_length=10.0,
        overall_mean_log_word_freq=3.0,
        slice_features=[slice_feat],
    )


class TestLoadModel:
    """Tests for _load_model function."""

    def test_load_model_success(
        self, tmp_path: Path, sample_model_spec: dict[str, Any]
    ):
        """Test successful model loading from valid JSON file."""
        model_file = tmp_path / "model.json"
        model_file.write_text(json.dumps(sample_model_spec), encoding="utf-8")

        with (
            patch.object(Path, "exists", return_value=True),
            patch(f"{ANALYZER_MODEL_PATH}.MODEL_PATH", model_file),
            patch(
                f"{ANALYZER_MODEL_PATH}.load_model_spec",
                return_value=sample_model_spec,
            ),
        ):
            # Clear the lru_cache to ensure fresh load
            _load_model.cache_clear()
            result = _load_model()

        assert result == sample_model_spec

    def test_load_model_file_not_found(self):
        """Test that FileNotFoundError is raised when model file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            _load_model.cache_clear()
            with pytest.raises(FileNotFoundError) as exc_info:
                _load_model()

        assert "Regression model spec not found" in str(exc_info.value)

    def test_load_model_caching(
        self, tmp_path: Path, sample_model_spec: dict[str, Any]
    ):
        """Test that model is cached after first load."""
        mock_load = MagicMock(return_value=sample_model_spec)

        with (
            patch.object(Path, "exists", return_value=True),
            patch(f"{ANALYZER_MODEL_PATH}.load_model_spec", mock_load),
        ):
            _load_model.cache_clear()
            # Call twice
            _load_model()
            _load_model()

        # Should only be called once due to caching
        assert mock_load.call_count == 1


class TestEstimateLexileFromFeatures:
    """Tests for estimate_lexile_from_features function."""

    def test_basic_estimation(
        self,
        sample_model_spec: dict[str, Any],
        sample_document_features: DocumentFeatures,
    ):
        """Test basic Lexile estimation with known coefficients."""
        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=sample_model_spec,
        ):
            result = estimate_lexile_from_features(sample_document_features)

        assert isinstance(result, float)
        # The result should be a reasonable Lexile value
        assert -500 <= result <= 2000

    def test_estimation_uses_all_features(
        self,
        sample_model_spec: dict[str, Any],
        sample_document_features: DocumentFeatures,
    ):
        """Test that estimation uses all expected features."""
        # Create a model spec that makes it easy to verify
        simple_spec = {
            "features": ["overall_msl", "overall_mlf"],
            "coefficients": [10.0, 20.0],
            "intercept": 100.0,
        }

        slice_feat = SliceFeatures(
            slice_id=0,
            num_tokens=50,
            num_sentences=5,
            mean_sentence_length=10.0,
            mean_log_word_freq=5.0,
        )
        features = DocumentFeatures(
            num_slices=1,
            total_tokens=50,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=5.0,
            slice_features=[slice_feat],
        )

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=simple_spec,
        ):
            result = estimate_lexile_from_features(features)

        # Expected: 100 + (10 * 10.0) + (5.0 * 20.0) = 100 + 100 + 100 = 300
        assert result == pytest.approx(300.0)

    def test_estimation_with_intercept_only(
        self, sample_document_features: DocumentFeatures
    ):
        """Test estimation when only intercept matters (zero coefficients)."""
        spec = {
            "features": ["overall_msl"],
            "coefficients": [0.0],
            "intercept": 500.0,
        }

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=spec,
        ):
            result = estimate_lexile_from_features(sample_document_features)

        assert result == pytest.approx(500.0)

    def test_estimation_with_missing_feature(
        self, sample_document_features: DocumentFeatures
    ):
        """Test estimation when model expects a feature not in feature dict."""
        spec = {
            "features": ["overall_msl", "nonexistent_feature"],
            "coefficients": [10.0, 100.0],
            "intercept": 200.0,
        }

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=spec,
        ):
            result = estimate_lexile_from_features(sample_document_features)

        # nonexistent_feature defaults to 0.0
        # Expected: 200 + (10 * 10.0) + (100 * 0.0) = 300
        assert result == pytest.approx(300.0)

    def test_estimation_mismatch_raises_error(
        self, sample_document_features: DocumentFeatures
    ):
        """Test that mismatched coefficients/features raises ValueError."""
        spec = {
            "features": ["overall_msl", "overall_mlf"],
            "coefficients": [10.0],  # Only one coefficient for two features
            "intercept": 100.0,
        }

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=spec,
        ):
            with pytest.raises(ValueError) as exc_info:
                estimate_lexile_from_features(sample_document_features)

        assert "coefficients/features mismatch" in str(exc_info.value)

    def test_estimation_with_negative_coefficients(
        self, sample_document_features: DocumentFeatures
    ):
        """Test estimation with negative coefficients."""
        spec = {
            "features": ["overall_msl"],
            "coefficients": [-10.0],
            "intercept": 500.0,
        }

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=spec,
        ):
            result = estimate_lexile_from_features(sample_document_features)

        # Expected: 500 + (-10 * 10.0) = 400
        assert result == pytest.approx(400.0)

    def test_estimation_returns_float(
        self,
        sample_model_spec: dict[str, Any],
        sample_document_features: DocumentFeatures,
    ):
        """Test that estimation always returns a float."""
        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=sample_model_spec,
        ):
            result = estimate_lexile_from_features(sample_document_features)

        assert type(result) is float

    def test_estimation_with_empty_features_list(self):
        """Test estimation with document having empty slice features."""
        spec = {
            "features": ["msl_std", "mlf_std"],
            "coefficients": [10.0, 20.0],
            "intercept": 500.0,
        }

        features = DocumentFeatures(
            num_slices=1,
            total_tokens=50,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=5.0,
            slice_features=[],  # Empty slice features affects std calculations
        )

        with patch(
            f"{ANALYZER_MODEL_PATH}._load_model",
            return_value=spec,
        ):
            result = estimate_lexile_from_features(features)

        # With empty slice_features, std values will be 0
        # Expected: 500 + (10 * 0) + (20 * 0) = 500
        assert result == pytest.approx(500.0)


class TestModelPath:
    """Tests for MODEL_PATH constant."""

    def test_model_path_is_path_object(self):
        """Test that MODEL_PATH is a Path object."""
        assert isinstance(MODEL_PATH, Path)

    def test_model_path_location(self):
        """Test that MODEL_PATH points to expected location."""
        assert "model" in str(MODEL_PATH)
        assert str(MODEL_PATH).endswith(".json")

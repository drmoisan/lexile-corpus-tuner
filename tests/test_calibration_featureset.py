# pyright: reportUnknownMemberType=false
"""Tests for calibration featureset module.

This module tests the make_regression_features function that converts
DocumentFeatures into a feature dictionary for regression.
"""

import math

import pytest
from lexile_corpus_tuner.lexile_scoring_model.analyzer.features import (
    DocumentFeatures,
    SliceFeatures,
)
from lexile_corpus_tuner.lexile_scoring_model.calibration.featureset import (
    make_regression_features,
)


@pytest.fixture
def sample_slice_features() -> list[SliceFeatures]:
    """Create sample slice features for testing."""
    return [
        SliceFeatures(
            slice_id=0,
            num_tokens=50,
            num_sentences=5,
            mean_sentence_length=10.0,
            mean_log_word_freq=3.0,
        ),
        SliceFeatures(
            slice_id=1,
            num_tokens=60,
            num_sentences=6,
            mean_sentence_length=10.0,
            mean_log_word_freq=4.0,
        ),
    ]


@pytest.fixture
def sample_document_features(
    sample_slice_features: list[SliceFeatures],
) -> DocumentFeatures:
    """Create sample document features for testing."""
    return DocumentFeatures(
        num_slices=2,
        total_tokens=110,
        overall_mean_sentence_length=10.0,
        overall_mean_log_word_freq=3.5,
        slice_features=sample_slice_features,
    )


class TestMakeRegressionFeatures:
    """Tests for make_regression_features function."""

    def test_returns_dict(self, sample_document_features: DocumentFeatures):
        """Test that function returns a dictionary."""
        result = make_regression_features(sample_document_features)
        assert isinstance(result, dict)

    def test_contains_basic_features(self, sample_document_features: DocumentFeatures):
        """Test that result contains basic features."""
        result = make_regression_features(sample_document_features)

        assert "overall_msl" in result
        assert "overall_mlf" in result
        assert "num_tokens" in result
        assert "num_slices" in result

    def test_overall_msl_value(self, sample_document_features: DocumentFeatures):
        """Test that overall_msl is correctly set."""
        result = make_regression_features(sample_document_features)
        assert result["overall_msl"] == pytest.approx(10.0)

    def test_overall_mlf_value(self, sample_document_features: DocumentFeatures):
        """Test that overall_mlf is correctly set."""
        result = make_regression_features(sample_document_features)
        assert result["overall_mlf"] == pytest.approx(3.5)

    def test_num_tokens_value(self, sample_document_features: DocumentFeatures):
        """Test that num_tokens is correctly set."""
        result = make_regression_features(sample_document_features)
        assert result["num_tokens"] == pytest.approx(110.0)

    def test_num_slices_value(self, sample_document_features: DocumentFeatures):
        """Test that num_slices is correctly set."""
        result = make_regression_features(sample_document_features)
        assert result["num_slices"] == pytest.approx(2.0)

    def test_log_num_tokens(self, sample_document_features: DocumentFeatures):
        """Test that log_num_tokens is calculated correctly."""
        result = make_regression_features(sample_document_features)
        expected = math.log(110.0)
        assert result["log_num_tokens"] == pytest.approx(expected)

    def test_log_num_tokens_with_zero_tokens(self):
        """Test that log_num_tokens handles zero tokens gracefully."""
        doc = DocumentFeatures(
            num_slices=0,
            total_tokens=0,
            overall_mean_sentence_length=0.0,
            overall_mean_log_word_freq=0.0,
            slice_features=[],
        )
        result = make_regression_features(doc)
        # log(max(1.0, 0)) = log(1.0) = 0.0
        assert result["log_num_tokens"] == pytest.approx(0.0)

    def test_msl_std_with_multiple_slices(
        self, sample_document_features: DocumentFeatures
    ):
        """Test that msl_std is calculated for multiple slices."""
        result = make_regression_features(sample_document_features)
        # Both slices have mean_sentence_length=10.0, so std=0
        assert result["msl_std"] == pytest.approx(0.0)

    def test_msl_std_with_varying_values(self):
        """Test msl_std calculation with varying values."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=50,
                num_sentences=5,
                mean_sentence_length=8.0,
                mean_log_word_freq=3.0,
            ),
            SliceFeatures(
                slice_id=1,
                num_tokens=60,
                num_sentences=6,
                mean_sentence_length=12.0,
                mean_log_word_freq=4.0,
            ),
        ]
        doc = DocumentFeatures(
            num_slices=2,
            total_tokens=110,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=3.5,
            slice_features=slices,
        )
        result = make_regression_features(doc)
        # pstdev of [8.0, 12.0] = sqrt(((8-10)^2 + (12-10)^2) / 2) = sqrt(4) = 2.0
        assert result["msl_std"] == pytest.approx(2.0)

    def test_msl_std_with_single_slice(self):
        """Test that msl_std is 0 for single slice."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=50,
                num_sentences=5,
                mean_sentence_length=10.0,
                mean_log_word_freq=3.0,
            ),
        ]
        doc = DocumentFeatures(
            num_slices=1,
            total_tokens=50,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=3.0,
            slice_features=slices,
        )
        result = make_regression_features(doc)
        assert result["msl_std"] == pytest.approx(0.0)

    def test_mlf_std_with_varying_values(self):
        """Test mlf_std calculation with varying values."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=50,
                num_sentences=5,
                mean_sentence_length=10.0,
                mean_log_word_freq=2.0,
            ),
            SliceFeatures(
                slice_id=1,
                num_tokens=60,
                num_sentences=6,
                mean_sentence_length=10.0,
                mean_log_word_freq=4.0,
            ),
        ]
        doc = DocumentFeatures(
            num_slices=2,
            total_tokens=110,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=3.0,
            slice_features=slices,
        )
        result = make_regression_features(doc)
        # pstdev of [2.0, 4.0] = sqrt(((2-3)^2 + (4-3)^2) / 2) = sqrt(1) = 1.0
        assert result["mlf_std"] == pytest.approx(1.0)

    def test_mlf_std_with_single_slice(self):
        """Test that mlf_std is 0 for single slice."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=50,
                num_sentences=5,
                mean_sentence_length=10.0,
                mean_log_word_freq=3.0,
            ),
        ]
        doc = DocumentFeatures(
            num_slices=1,
            total_tokens=50,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=3.0,
            slice_features=slices,
        )
        result = make_regression_features(doc)
        assert result["mlf_std"] == pytest.approx(0.0)

    def test_overall_msl_sq(self, sample_document_features: DocumentFeatures):
        """Test that overall_msl_sq is calculated correctly."""
        result = make_regression_features(sample_document_features)
        expected = 10.0**2
        assert result["overall_msl_sq"] == pytest.approx(expected)

    def test_overall_mlf_sq(self, sample_document_features: DocumentFeatures):
        """Test that overall_mlf_sq is calculated correctly."""
        result = make_regression_features(sample_document_features)
        expected = 3.5**2
        assert result["overall_mlf_sq"] == pytest.approx(expected)

    def test_msl_times_mlf(self, sample_document_features: DocumentFeatures):
        """Test that msl_times_mlf is calculated correctly."""
        result = make_regression_features(sample_document_features)
        expected = 10.0 * 3.5
        assert result["msl_times_mlf"] == pytest.approx(expected)

    def test_all_values_are_float(self, sample_document_features: DocumentFeatures):
        """Test that all returned values are floats."""
        result = make_regression_features(sample_document_features)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_empty_slice_features(self):
        """Test handling of empty slice features list."""
        doc = DocumentFeatures(
            num_slices=0,
            total_tokens=0,
            overall_mean_sentence_length=0.0,
            overall_mean_log_word_freq=0.0,
            slice_features=[],
        )
        result = make_regression_features(doc)

        assert result["msl_std"] == pytest.approx(0.0)
        assert result["mlf_std"] == pytest.approx(0.0)
        assert result["overall_msl_sq"] == pytest.approx(0.0)
        assert result["overall_mlf_sq"] == pytest.approx(0.0)
        assert result["msl_times_mlf"] == pytest.approx(0.0)

    def test_large_values(self):
        """Test handling of large feature values."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=100000,
                num_sentences=1000,
                mean_sentence_length=100.0,
                mean_log_word_freq=10.0,
            ),
        ]
        doc = DocumentFeatures(
            num_slices=1,
            total_tokens=100000,
            overall_mean_sentence_length=100.0,
            overall_mean_log_word_freq=10.0,
            slice_features=slices,
        )
        result = make_regression_features(doc)

        assert result["num_tokens"] == pytest.approx(100000.0)
        assert result["log_num_tokens"] == pytest.approx(math.log(100000.0))
        assert result["overall_msl_sq"] == pytest.approx(10000.0)
        assert result["overall_mlf_sq"] == pytest.approx(100.0)
        assert result["msl_times_mlf"] == pytest.approx(1000.0)

    def test_negative_mean_log_word_freq(self):
        """Test handling of negative mean_log_word_freq (rare words)."""
        slices = [
            SliceFeatures(
                slice_id=0,
                num_tokens=50,
                num_sentences=5,
                mean_sentence_length=10.0,
                mean_log_word_freq=-2.0,  # Very rare words
            ),
        ]
        doc = DocumentFeatures(
            num_slices=1,
            total_tokens=50,
            overall_mean_sentence_length=10.0,
            overall_mean_log_word_freq=-2.0,
            slice_features=slices,
        )
        result = make_regression_features(doc)

        assert result["overall_mlf"] == pytest.approx(-2.0)
        assert result["overall_mlf_sq"] == pytest.approx(4.0)
        assert result["msl_times_mlf"] == pytest.approx(-20.0)

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedVariable=false
"""Tests for calibration train module.

This module tests the training workflow for the Lexile regression model,
including metrics computation and model training.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from lexile_corpus_tuner.lexile_scoring_model.calibration.train import (
    FEATURE_COLS,
    TARGET_COL,
    compute_metrics,
    train_regression_model,
)


@pytest.fixture
def sample_calibration_df() -> pd.DataFrame:
    """Create a sample calibration dataframe for testing."""
    np.random.seed(42)  # For reproducibility
    n_samples = 100

    data: dict[str, Any] = {
        "overall_msl": np.random.uniform(5.0, 20.0, n_samples),
        "overall_mlf": np.random.uniform(2.0, 5.0, n_samples),
        "log_num_tokens": np.log(np.random.uniform(100, 10000, n_samples)),
        "msl_std": np.random.uniform(0.0, 5.0, n_samples),
        "mlf_std": np.random.uniform(0.0, 1.0, n_samples),
        "overall_msl_sq": np.zeros(n_samples),  # Will be filled
        "overall_mlf_sq": np.zeros(n_samples),  # Will be filled
        "msl_times_mlf": np.zeros(n_samples),  # Will be filled
        "num_tokens": np.random.uniform(100, 10000, n_samples),
        "num_slices": np.random.randint(1, 10, n_samples),
    }
    data["overall_msl_sq"] = data["overall_msl"] ** 2
    data["overall_mlf_sq"] = data["overall_mlf"] ** 2
    data["msl_times_mlf"] = data["overall_msl"] * data["overall_mlf"]
    # Target: Lexile correlates with sentence length and word frequency
    data["lexile_official"] = (
        500
        + 20 * data["overall_msl"]
        - 100 * data["overall_mlf"]
        + np.random.normal(0, 20, n_samples)
    )

    return pd.DataFrame(data)


@pytest.fixture
def minimal_calibration_df() -> pd.DataFrame:
    """Create a minimal calibration dataframe (< 5 samples)."""
    data: dict[str, Any] = {
        "overall_msl": [10.0, 12.0, 8.0],
        "overall_mlf": [3.5, 4.0, 3.0],
        "log_num_tokens": [5.0, 5.5, 4.5],
        "msl_std": [1.0, 1.5, 0.5],
        "mlf_std": [0.5, 0.6, 0.4],
        "overall_msl_sq": [100.0, 144.0, 64.0],
        "overall_mlf_sq": [12.25, 16.0, 9.0],
        "msl_times_mlf": [35.0, 48.0, 24.0],
        "num_tokens": [500, 600, 400],
        "num_slices": [2, 3, 2],
        "lexile_official": [500, 600, 450],
    }
    return pd.DataFrame(data)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        result = compute_metrics(y_true, y_pred)

        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["r"] == pytest.approx(1.0)

    def test_constant_offset(self):
        """Test metrics with constant offset in predictions."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([110.0, 210.0, 310.0, 410.0, 510.0])  # +10 offset

        result = compute_metrics(y_true, y_pred)

        assert result["rmse"] == pytest.approx(10.0)
        assert result["mae"] == pytest.approx(10.0)
        # Correlation should be 1.0 since prediction is linear transform
        assert result["r"] == pytest.approx(1.0)

    def test_single_value(self):
        """Test metrics with single value returns nan for correlation."""
        y_true = np.array([100.0])
        y_pred = np.array([110.0])

        result = compute_metrics(y_true, y_pred)

        assert result["rmse"] == pytest.approx(10.0)
        assert result["mae"] == pytest.approx(10.0)
        assert np.isnan(result["r"])

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([150.0, 180.0, 320.0, 380.0, 520.0])

        result = compute_metrics(y_true, y_pred)

        assert result["rmse"] > 0
        assert result["mae"] > 0
        assert -1.0 <= result["r"] <= 1.0

    def test_negative_correlation(self):
        """Test metrics with negatively correlated predictions."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([500.0, 400.0, 300.0, 200.0, 100.0])  # Inverted

        result = compute_metrics(y_true, y_pred)

        assert result["r"] == pytest.approx(-1.0)

    def test_rmse_larger_than_mae(self):
        """Test that RMSE is always >= MAE."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = np.array([110.0, 190.0, 350.0, 390.0, 480.0])

        result = compute_metrics(y_true, y_pred)

        assert result["rmse"] >= result["mae"]

    def test_returns_float_values(self):
        """Test that all returned values are Python floats."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])

        result = compute_metrics(y_true, y_pred)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"


class TestTrainRegressionModel:
    """Tests for train_regression_model function."""

    def test_returns_model_and_metrics(self, sample_calibration_df: pd.DataFrame):
        """Test that function returns model and metrics tuple."""
        model, metrics = train_regression_model(sample_calibration_df)

        assert model is not None
        assert isinstance(metrics, dict)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r" in metrics

    def test_model_has_coefficients(self, sample_calibration_df: pd.DataFrame):
        """Test that trained model has coefficients."""
        model, _ = train_regression_model(sample_calibration_df)

        # Cast to access sklearn attributes
        model_any = cast("Any", model)
        assert hasattr(model_any, "coef_")
        assert len(model_any.coef_) == len(FEATURE_COLS)

    def test_model_has_intercept(self, sample_calibration_df: pd.DataFrame):
        """Test that trained model has intercept."""
        model, _ = train_regression_model(sample_calibration_df)

        model_any = cast("Any", model)
        assert hasattr(model_any, "intercept_")

    def test_model_can_predict(self, sample_calibration_df: pd.DataFrame):
        """Test that trained model can make predictions."""
        model, _ = train_regression_model(sample_calibration_df)

        # Create sample features for prediction
        X_sample = np.array([[10.0, 3.5, 5.0, 1.0, 0.5, 100.0, 12.25, 35.0]])
        model_any = cast("Any", model)
        predictions = model_any.predict(X_sample)

        assert len(predictions) == 1
        assert isinstance(predictions[0], int | float | np.floating)

    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValueError."""
        all_cols = FEATURE_COLS + [TARGET_COL, "num_tokens", "num_slices"]
        empty_df = pd.DataFrame(columns=all_cols)

        with pytest.raises(ValueError) as exc_info:
            train_regression_model(empty_df)

        assert "empty after filtering" in str(exc_info.value).lower()

    def test_minimal_dataset(self, minimal_calibration_df: pd.DataFrame):
        """Test training with minimal dataset (< 5 samples)."""
        model, metrics = train_regression_model(minimal_calibration_df)

        assert model is not None
        assert isinstance(metrics, dict)

    def test_filters_by_num_tokens(self, sample_calibration_df: pd.DataFrame):
        """Test that documents with < 100 tokens are filtered."""
        # Add some rows with low token counts
        df = sample_calibration_df.copy()
        low_token_rows = df.iloc[:5].copy()
        low_token_rows["num_tokens"] = 50  # Below 100 threshold
        df = pd.concat([df, low_token_rows], ignore_index=True)

        # Should not raise error, but should filter those rows
        model, metrics = train_regression_model(df)
        assert model is not None

    def test_filters_by_num_slices(self, sample_calibration_df: pd.DataFrame):
        """Test that documents with < 1 slice are filtered."""
        df = sample_calibration_df.copy()
        zero_slice_rows = df.iloc[:5].copy()
        zero_slice_rows["num_slices"] = 0  # Below 1 threshold
        df = pd.concat([df, zero_slice_rows], ignore_index=True)

        # Should not raise error, but should filter those rows
        model, metrics = train_regression_model(df)
        assert model is not None

    def test_handles_nan_values(self, sample_calibration_df: pd.DataFrame):
        """Test that rows with NaN values are filtered."""
        df = sample_calibration_df.copy()
        # Set some values to NaN
        df.iloc[0, df.columns.get_loc("overall_msl")] = np.nan
        df.iloc[1, df.columns.get_loc("lexile_official")] = np.nan

        model, metrics = train_regression_model(df)
        assert model is not None

    def test_reproducible_with_seed(self, sample_calibration_df: pd.DataFrame):
        """Test that training is reproducible due to random_state."""
        model1, metrics1 = train_regression_model(sample_calibration_df)
        model2, metrics2 = train_regression_model(sample_calibration_df)

        model1_any = cast("Any", model1)
        model2_any = cast("Any", model2)

        # Coefficients should be identical
        np.testing.assert_array_almost_equal(model1_any.coef_, model2_any.coef_)
        assert model1_any.intercept_ == pytest.approx(model2_any.intercept_)

    def test_metrics_reasonable_values(self, sample_calibration_df: pd.DataFrame):
        """Test that metrics have reasonable values."""
        _, metrics = train_regression_model(sample_calibration_df)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"]
        assert -1.0 <= metrics["r"] <= 1.0


class TestFeatureAndTargetConstants:
    """Tests for FEATURE_COLS and TARGET_COL constants."""

    def test_feature_cols_is_list(self):
        """Test that FEATURE_COLS is a list."""
        assert isinstance(FEATURE_COLS, list)

    def test_feature_cols_not_empty(self):
        """Test that FEATURE_COLS is not empty."""
        assert len(FEATURE_COLS) > 0

    def test_feature_cols_contains_expected_features(self):
        """Test that FEATURE_COLS contains expected feature names."""
        expected = {
            "overall_msl",
            "overall_mlf",
            "log_num_tokens",
            "msl_std",
            "mlf_std",
            "overall_msl_sq",
            "overall_mlf_sq",
            "msl_times_mlf",
        }
        assert set(FEATURE_COLS) == expected

    def test_target_col_is_string(self):
        """Test that TARGET_COL is a string."""
        assert isinstance(TARGET_COL, str)

    def test_target_col_value(self):
        """Test that TARGET_COL has expected value."""
        assert TARGET_COL == "lexile_official"

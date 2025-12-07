"""Tests for calibration model_store module.

This module tests model serialization and deserialization functionality
for the calibration workflow. Tests use mocked sklearn models and tmp_path
for file operations.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from lexile_corpus_tuner.lexile_scoring_model.calibration.model_store import (
    load_model_spec,
    save_model,
)


class TestSaveModel:
    """Tests for save_model function."""

    @pytest.fixture
    def mock_elastic_net_model(self) -> MagicMock:
        """Create a mock ElasticNet model with coefficients and intercept."""
        model = MagicMock()
        model.coef_ = [0.1, 0.2, 0.3]
        model.intercept_ = 1.5
        return model

    @pytest.fixture
    def sample_metrics(self) -> dict[str, float]:
        """Create sample metrics dictionary."""
        return {"rmse": 25.5, "mae": 18.2, "r": 0.85}

    @pytest.fixture
    def sample_feature_names(self) -> list[str]:
        """Create sample feature names list."""
        return ["feature_a", "feature_b", "feature_c"]

    def test_save_model_creates_json_file(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that save_model creates a JSON file at the specified path."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        assert output_path.exists()
        assert output_path.is_file()

    def test_save_model_spec_contains_version(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved model spec contains today's date as version."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["version"] == date.today().isoformat()

    def test_save_model_spec_contains_features(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved model spec contains feature names."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["features"] == sample_feature_names

    def test_save_model_spec_contains_coefficients(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved model spec contains model coefficients as floats."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["coefficients"] == [0.1, 0.2, 0.3]
        assert all(isinstance(c, float) for c in spec["coefficients"])

    def test_save_model_spec_contains_intercept(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved model spec contains model intercept as float."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["intercept"] == 1.5
        assert isinstance(spec["intercept"], float)

    def test_save_model_spec_contains_metrics(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved model spec contains provided metrics."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["metrics"] == sample_metrics

    def test_save_model_creates_parent_directories(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that save_model creates parent directories if they don't exist."""
        # Arrange
        nested_path = tmp_path / "deep" / "nested" / "dir" / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, nested_path
        )

        # Assert
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_model_with_numpy_array_coefficients(
        self,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that save_model handles numpy array coefficients correctly."""
        # Arrange
        import numpy as np

        model = MagicMock()
        model.coef_ = np.array([0.5, 0.6, 0.7])
        model.intercept_ = np.float64(2.5)
        output_path = tmp_path / "model.json"

        # Act
        save_model(model, sample_metrics, sample_feature_names, output_path)

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert spec["coefficients"] == [0.5, 0.6, 0.7]
        assert spec["intercept"] == 2.5

    def test_save_model_overwrites_existing_file(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that save_model overwrites an existing file."""
        # Arrange
        output_path = tmp_path / "model.json"
        output_path.write_text('{"old": "data"}', encoding="utf-8")

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        spec = json.loads(output_path.read_text(encoding="utf-8"))
        assert "old" not in spec
        assert "version" in spec

    def test_save_model_json_is_indented(
        self,
        mock_elastic_net_model: MagicMock,
        sample_metrics: dict[str, float],
        sample_feature_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test that saved JSON is formatted with indentation."""
        # Arrange
        output_path = tmp_path / "model.json"

        # Act
        save_model(
            mock_elastic_net_model, sample_metrics, sample_feature_names, output_path
        )

        # Assert
        content = output_path.read_text(encoding="utf-8")
        # Indented JSON has newlines and spaces
        assert "\n" in content
        assert "  " in content


class TestLoadModelSpec:
    """Tests for load_model_spec function."""

    def test_load_model_spec_returns_dict(self, tmp_path: Path) -> None:
        """Test that load_model_spec returns a dictionary."""
        # Arrange
        model_path = tmp_path / "model.json"
        spec: dict[str, Any] = {
            "version": "2024-01-15",
            "features": ["a", "b"],
            "coefficients": [0.1, 0.2],
            "intercept": 1.0,
            "metrics": {"rmse": 20.0},
        }
        model_path.write_text(json.dumps(spec), encoding="utf-8")

        # Act
        result = load_model_spec(model_path)

        # Assert
        assert isinstance(result, dict)
        assert result == spec

    def test_load_model_spec_preserves_all_fields(self, tmp_path: Path) -> None:
        """Test that load_model_spec preserves all fields from the file."""
        # Arrange
        model_path = tmp_path / "model.json"
        spec: dict[str, Any] = {
            "version": "2024-01-15",
            "features": ["feature_x", "feature_y", "feature_z"],
            "coefficients": [1.1, 2.2, 3.3],
            "intercept": 4.4,
            "metrics": {"rmse": 15.0, "mae": 12.0, "r": 0.92},
        }
        model_path.write_text(json.dumps(spec), encoding="utf-8")

        # Act
        result = load_model_spec(model_path)

        # Assert
        assert result["version"] == "2024-01-15"
        assert result["features"] == ["feature_x", "feature_y", "feature_z"]
        assert result["coefficients"] == [1.1, 2.2, 3.3]
        assert result["intercept"] == 4.4
        assert result["metrics"]["rmse"] == 15.0

    def test_load_model_spec_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Test that load_model_spec raises FileNotFoundError for missing file."""
        # Arrange
        nonexistent_path = tmp_path / "nonexistent.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_model_spec(nonexistent_path)

    def test_load_model_spec_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Test that load_model_spec raises JSONDecodeError for invalid JSON."""
        # Arrange
        model_path = tmp_path / "invalid.json"
        model_path.write_text("not valid json {", encoding="utf-8")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            load_model_spec(model_path)

    def test_load_model_spec_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Test that load_model_spec raises error for empty file."""
        # Arrange
        model_path = tmp_path / "empty.json"
        model_path.write_text("", encoding="utf-8")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            load_model_spec(model_path)

    def test_load_model_spec_reads_utf8_encoding(self, tmp_path: Path) -> None:
        """Test that load_model_spec correctly reads UTF-8 encoded content."""
        # Arrange
        model_path = tmp_path / "model.json"
        spec: dict[str, Any] = {
            "version": "2024-01-15",
            "features": ["fëature", "résumé"],
            "coefficients": [0.1, 0.2],
            "intercept": 1.0,
            "metrics": {},
        }
        model_path.write_text(json.dumps(spec), encoding="utf-8")

        # Act
        result = load_model_spec(model_path)

        # Assert
        assert result["features"] == ["fëature", "résumé"]


class TestSaveAndLoadModelRoundTrip:
    """Integration tests for save_model and load_model_spec together."""

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        """Test that saving and loading a model preserves all data."""
        # Arrange
        model = MagicMock()
        model.coef_ = [0.25, 0.50, 0.75]
        model.intercept_ = 2.0
        metrics = {"rmse": 22.5, "mae": 17.3, "r": 0.88}
        feature_names = ["feat_1", "feat_2", "feat_3"]
        output_path = tmp_path / "roundtrip_model.json"

        # Act
        save_model(model, metrics, feature_names, output_path)
        loaded_spec = load_model_spec(output_path)

        # Assert
        assert loaded_spec["features"] == feature_names
        assert loaded_spec["coefficients"] == [0.25, 0.50, 0.75]
        assert loaded_spec["intercept"] == 2.0
        assert loaded_spec["metrics"] == metrics
        assert loaded_spec["version"] == date.today().isoformat()

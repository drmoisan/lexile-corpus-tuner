"""Tests for analyzer CLI module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result
from pytest import MonkeyPatch

from lexile_corpus_tuner.analyzer.cli import analyze_group


class TestAnalyzerCLI:
    """Tests for analyzer CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_analyzer_components(self, monkeypatch: MonkeyPatch):
        """Mock all analyzer components."""
        mock_build_slices = MagicMock()
        mock_compute_features = MagicMock()
        mock_estimate = MagicMock()
        mock_adjust = MagicMock()

        monkeypatch.setattr(
            "lexile_corpus_tuner.analyzer.cli.build_slices", mock_build_slices
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.analyzer.cli.compute_document_features",
            mock_compute_features,
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.analyzer.cli.estimate_lexile_from_features",
            mock_estimate,
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.analyzer.cli.adjust_for_special_cases", mock_adjust
        )

        # Setup default return values
        mock_features = MagicMock()
        mock_features.num_slices = 1
        mock_features.total_tokens = 100
        mock_features.overall_mean_sentence_length = 10.0
        mock_features.overall_mean_log_word_freq = 3.5
        mock_features.slice_features = []
        mock_compute_features.return_value = mock_features

        mock_estimate.return_value = 500.0
        mock_adjust.return_value = 500.0

        return {
            "build_slices": mock_build_slices,
            "compute_features": mock_compute_features,
            "estimate": mock_estimate,
            "adjust": mock_adjust,
        }

    def test_analyze_text_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_components: dict[str, MagicMock],
    ):
        """Test basic text analysis."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Sample text.", encoding="utf-8")

        result: Result = runner.invoke(analyze_group, ["text", str(input_file)])

        assert result.exit_code == 0
        assert "File: " in result.output
        assert "Estimated Lexile (raw): 500.0L" in result.output

        mock_analyzer_components["build_slices"].assert_called_once()
        mock_analyzer_components["compute_features"].assert_called_once()
        mock_analyzer_components["estimate"].assert_called_once()
        mock_analyzer_components["adjust"].assert_called_once_with(
            500.0, is_picture_book=False, is_emergent_nonfiction=False
        )

    def test_analyze_text_json_output(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_components: dict[str, MagicMock],
    ):
        """Test text analysis with JSON output."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Sample text.", encoding="utf-8")
        output_file = tmp_path / "output.json"

        result: Result = runner.invoke(
            analyze_group, ["text", str(input_file), "--json-output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert data["raw_lexile"] == 500.0
        assert data["adjusted_lexile"] == 500.0
        assert data["document_features"]["total_tokens"] == 100

    def test_analyze_text_flags(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_components: dict[str, MagicMock],
    ):
        """Test text analysis with adjustment flags."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Sample text.", encoding="utf-8")

        result: Result = runner.invoke(
            analyze_group,
            ["text", str(input_file), "--picture-book", "--emergent-nonfiction"],
        )

        assert result.exit_code == 0
        mock_analyzer_components["adjust"].assert_called_once_with(
            500.0, is_picture_book=True, is_emergent_nonfiction=True
        )

    def test_analyze_text_missing_file(self, runner: CliRunner, tmp_path: Path):
        """Test analysis with missing file."""
        input_file = tmp_path / "missing.txt"

        result: Result = runner.invoke(analyze_group, ["text", str(input_file)])

        assert result.exit_code != 0
        assert "does not exist" in result.output

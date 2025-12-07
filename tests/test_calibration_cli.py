"""Tests for calibration CLI module.

This module tests the CLI commands for the calibration workflow,
including fetch-texts, build-dataset, and fit commands.
All HTTP requests and file system operations use mocks and tmp_path.

Note: This file tests some private helper functions (prefixed with _) because
they contain significant logic that benefits from direct testing.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner, Result

CALIB_CLI_PATH = "lexile_corpus_tuner.lexile_scoring_model.calibration.cli"


class TestCalibrationGroup:
    """Tests for calibration CLI command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_calibration_group_exists(self, runner: CliRunner) -> None:
        """Test that calibration group command exists."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(calibration_group, ["--help"])
        assert result.exit_code == 0

    def test_calibration_group_has_subcommands(self, runner: CliRunner) -> None:
        """Test that calibration group has expected subcommands."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(calibration_group, ["--help"])
        assert "fetch-texts" in result.output
        assert "build-dataset" in result.output
        assert "fit" in result.output


class TestFetchTexts:
    """Tests for fetch-texts command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_catalog(self, tmp_path: Path) -> Path:
        """Create a sample catalog CSV file."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "text1,350L,manual,\n"
            "text2,400L,http,http://example.com/text.txt\n",
            encoding="utf-8",
        )
        return catalog

    def test_fetch_texts_requires_catalog_option(self, runner: CliRunner) -> None:
        """Test that fetch-texts requires --catalog option."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(calibration_group, ["fetch-texts"])
        assert result.exit_code != 0
        assert "catalog" in result.output.lower() or "missing" in result.output.lower()

    def test_fetch_texts_requires_texts_root_option(
        self, runner: CliRunner, sample_catalog: Path
    ) -> None:
        """Test that fetch-texts requires --texts-root option."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            ["fetch-texts", "--catalog", str(sample_catalog)],
        )
        assert result.exit_code != 0

    def test_fetch_texts_creates_texts_directory(
        self, runner: CliRunner, sample_catalog: Path, tmp_path: Path
    ) -> None:
        """Test that fetch-texts creates the texts directory."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        texts_dir = tmp_path / "texts"

        result: Result = runner.invoke(
            calibration_group,
            [
                "fetch-texts",
                "--catalog",
                str(sample_catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )

        assert result.exit_code == 0
        assert texts_dir.exists()

    def test_fetch_texts_manual_acquisition_reports_pending(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that manual acquisition texts are reported as pending."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "manual_text,350L,manual,\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "fetch-texts",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )

        assert result.exit_code == 0
        assert "pending" in result.output.lower()

    def test_fetch_texts_skips_existing_files(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that fetch-texts skips already existing text files."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "existing_text,350L,http,http://example.com/text.txt\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "existing_text.txt").write_text(
            "existing content", encoding="utf-8"
        )

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "fetch-texts",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )

        assert result.exit_code == 0
        assert "1 already existed" in result.output

    def test_fetch_texts_http_acquisition_success(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that HTTP acquisition downloads text."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "http_text,350L,http,http://example.com/text.txt\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with patch(f"{CALIB_CLI_PATH}._fetch_http_text") as mock_fetch:
            mock_fetch.return_value = None  # Function writes to file, returns None

            # Make fetch_http_text create the file
            def write_text_file(url: str, dest: Path) -> None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("downloaded content", encoding="utf-8")

            mock_fetch.side_effect = write_text_file

            result: Result = runner.invoke(
                calibration_group,
                [
                    "fetch-texts",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Downloaded 1" in result.output

    def test_fetch_texts_gutenberg_acquisition_success(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that Gutenberg acquisition downloads text."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "gutenberg_text,350L,gutenberg,12345\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with patch(f"{CALIB_CLI_PATH}._fetch_gutenberg_text") as mock_fetch:

            def write_text_file(ebook_id: str, dest: Path) -> None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("gutenberg content", encoding="utf-8")

            mock_fetch.side_effect = write_text_file

            result: Result = runner.invoke(
                calibration_group,
                [
                    "fetch-texts",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Downloaded 1" in result.output

    def test_fetch_texts_unsupported_acquisition_type(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that unsupported acquisition types are reported as failures."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "unknown_text,350L,ftp,ftp://example.com/text.txt\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "fetch-texts",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )

        assert result.exit_code == 0
        assert "1 failures" in result.output

    def test_fetch_texts_handles_exception(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that exceptions during fetch are handled gracefully."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            "error_text,350L,http,http://example.com/error.txt\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with patch(f"{CALIB_CLI_PATH}._fetch_http_text") as mock_fetch:
            mock_fetch.side_effect = RuntimeError("Network error")

            result: Result = runner.invoke(
                calibration_group,
                [
                    "fetch-texts",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                ],
            )

        assert result.exit_code == 0
        assert "1 failures" in result.output

    def test_fetch_texts_skips_rows_without_text_id(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that rows without text_id are skipped."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,acquisition_type,acquisition_key\n"
            ",350L,manual,\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "fetch-texts",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Downloaded 0" in result.output


class TestBuildDataset:
    """Tests for build-dataset command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_build_dataset_requires_catalog_option(self, runner: CliRunner) -> None:
        """Test that build-dataset requires --catalog option."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(calibration_group, ["build-dataset"])
        assert result.exit_code != 0

    def test_build_dataset_requires_texts_root_option(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that build-dataset requires --texts-root option."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text("text_id,lexile_official\n", encoding="utf-8")

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            ["build-dataset", "--catalog", str(catalog)],
        )
        assert result.exit_code != 0

    def test_build_dataset_requires_output_option(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that build-dataset requires --output option."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text("text_id,lexile_official\n", encoding="utf-8")
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "build-dataset",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
            ],
        )
        assert result.exit_code != 0

    def test_build_dataset_success_parquet(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test build-dataset creates parquet output."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title,author,grade_band,lexile_source\n"
            "text1,350L,Title 1,Author 1,3-5,Official\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "text1.txt").write_text(
            "This is a sample text with enough words to create slices. "
            "The cat sat on the mat. The dog ran in the park. "
            "Children played outside all day long. Birds flew in the sky.",
            encoding="utf-8",
        )
        output = tmp_path / "dataset.parquet"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        # Mock the feature computation to speed up tests
        with (
            patch(f"{CALIB_CLI_PATH}.build_slices") as mock_slices,
            patch(f"{CALIB_CLI_PATH}.compute_document_features") as mock_features,
            patch(f"{CALIB_CLI_PATH}.make_regression_features") as mock_regression,
        ):
            mock_slices.return_value = []
            mock_doc_features = MagicMock()
            mock_doc_features.total_tokens = 100
            mock_doc_features.num_slices = 1
            mock_doc_features.overall_mean_sentence_length = 10.0
            mock_doc_features.overall_mean_log_word_freq = 5.0
            mock_features.return_value = mock_doc_features
            mock_regression.return_value = {"feat1": 0.5, "feat2": 0.3}

            result: Result = runner.invoke(
                calibration_group,
                [
                    "build-dataset",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert output.exists()
        assert "Wrote 1 calibration rows" in result.output

    def test_build_dataset_success_csv(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test build-dataset creates CSV output."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title,author,grade_band,lexile_source\n"
            "text1,350L,Title 1,Author 1,3-5,Official\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "text1.txt").write_text(
            "This is a sample text.",
            encoding="utf-8",
        )
        output = tmp_path / "dataset.csv"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with (
            patch(f"{CALIB_CLI_PATH}.build_slices") as mock_slices,
            patch(f"{CALIB_CLI_PATH}.compute_document_features") as mock_features,
            patch(f"{CALIB_CLI_PATH}.make_regression_features") as mock_regression,
        ):
            mock_slices.return_value = []
            mock_doc_features = MagicMock()
            mock_doc_features.total_tokens = 100
            mock_doc_features.num_slices = 1
            mock_doc_features.overall_mean_sentence_length = 10.0
            mock_doc_features.overall_mean_log_word_freq = 5.0
            mock_features.return_value = mock_doc_features
            mock_regression.return_value = {"feat1": 0.5}

            result: Result = runner.invoke(
                calibration_group,
                [
                    "build-dataset",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert output.exists()

    def test_build_dataset_missing_text_non_strict(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test build-dataset skips missing texts in non-strict mode."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title\n"
            "missing_text,350L,Missing Title\n"
            "present_text,400L,Present Title\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "present_text.txt").write_text(
            "This is a sample text.",
            encoding="utf-8",
        )
        output = tmp_path / "dataset.parquet"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with (
            patch(f"{CALIB_CLI_PATH}.build_slices") as mock_slices,
            patch(f"{CALIB_CLI_PATH}.compute_document_features") as mock_features,
            patch(f"{CALIB_CLI_PATH}.make_regression_features") as mock_regression,
        ):
            mock_slices.return_value = []
            mock_doc_features = MagicMock()
            mock_doc_features.total_tokens = 100
            mock_doc_features.num_slices = 1
            mock_doc_features.overall_mean_sentence_length = 10.0
            mock_doc_features.overall_mean_log_word_freq = 5.0
            mock_features.return_value = mock_doc_features
            mock_regression.return_value = {}

            result: Result = runner.invoke(
                calibration_group,
                [
                    "build-dataset",
                    "--catalog",
                    str(catalog),
                    "--texts-root",
                    str(texts_dir),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "missing texts skipped: 1" in result.output

    def test_build_dataset_missing_text_strict(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test build-dataset fails on missing texts in strict mode."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official\n" "missing_text,350L\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        output = tmp_path / "dataset.parquet"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "build-dataset",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
                "--output",
                str(output),
                "--strict",
            ],
        )

        assert result.exit_code != 0
        assert "Missing text" in result.output

    def test_build_dataset_skips_invalid_lexile(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test build-dataset skips entries with invalid lexile values."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title\n" "invalid_lexile,invalid,Title\n",
            encoding="utf-8",
        )
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "invalid_lexile.txt").write_text(
            "This is a sample text.",
            encoding="utf-8",
        )
        output = tmp_path / "dataset.parquet"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "build-dataset",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
                "--output",
                str(output),
            ],
        )

        # Should fail because no valid rows were added
        assert result.exit_code != 0
        assert "No rows were added" in result.output

    def test_build_dataset_empty_catalog(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test build-dataset fails on empty catalog."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text("text_id,lexile_official\n", encoding="utf-8")
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        output = tmp_path / "dataset.parquet"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(
            calibration_group,
            [
                "build-dataset",
                "--catalog",
                str(catalog),
                "--texts-root",
                str(texts_dir),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code != 0
        assert "No rows were added" in result.output


class TestFit:
    """Tests for fit command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_fit_requires_dataset_argument(self, runner: CliRunner) -> None:
        """Test that fit requires dataset argument."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        result: Result = runner.invoke(calibration_group, ["fit"])
        assert result.exit_code != 0

    def test_fit_success_parquet_input(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test fit with parquet input dataset."""
        import pandas as pd

        dataset_path = tmp_path / "dataset.parquet"
        df = pd.DataFrame(
            {
                "lexile_official": [300.0, 400.0, 350.0],
                "log_mean_sentence_length": [2.0, 2.5, 2.2],
                "log_mean_word_length": [1.0, 1.2, 1.1],
            }
        )
        df.to_parquet(dataset_path, index=False)  # type: ignore[reportUnknownMemberType]
        output_path = tmp_path / "model.json"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with (
            patch(f"{CALIB_CLI_PATH}.train_regression_model") as mock_train,
            patch(f"{CALIB_CLI_PATH}.save_model") as mock_save,
        ):
            mock_model = MagicMock()
            mock_train.return_value = (
                mock_model,
                {"rmse": 25.0, "mae": 20.0, "r": 0.9},
            )

            result: Result = runner.invoke(
                calibration_group,
                ["fit", str(dataset_path), "--out", str(output_path)],
            )

        assert result.exit_code == 0
        assert "Saved model" in result.output
        mock_train.assert_called_once()
        mock_save.assert_called_once()

    def test_fit_success_csv_input(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test fit with CSV input dataset."""
        import pandas as pd

        dataset_path = tmp_path / "dataset.csv"
        df = pd.DataFrame(
            {
                "lexile_official": [300.0, 400.0, 350.0],
                "log_mean_sentence_length": [2.0, 2.5, 2.2],
                "log_mean_word_length": [1.0, 1.2, 1.1],
            }
        )
        df.to_csv(dataset_path, index=False)  # type: ignore[reportUnknownMemberType]
        output_path = tmp_path / "model.json"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with (
            patch(f"{CALIB_CLI_PATH}.train_regression_model") as mock_train,
            patch(f"{CALIB_CLI_PATH}.save_model"),
        ):
            mock_model = MagicMock()
            mock_train.return_value = (
                mock_model,
                {"rmse": 25.0, "mae": 20.0, "r": 0.9},
            )

            result: Result = runner.invoke(
                calibration_group,
                ["fit", str(dataset_path), "--out", str(output_path)],
            )

        assert result.exit_code == 0
        mock_train.assert_called_once()

    def test_fit_reports_metrics(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that fit reports validation metrics."""
        import pandas as pd

        dataset_path = tmp_path / "dataset.parquet"
        df = pd.DataFrame(
            {
                "lexile_official": [300.0, 400.0, 350.0],
                "log_mean_sentence_length": [2.0, 2.5, 2.2],
            }
        )
        df.to_parquet(dataset_path, index=False)  # type: ignore[reportUnknownMemberType]
        output_path = tmp_path / "model.json"

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            calibration_group,
        )

        with (
            patch(f"{CALIB_CLI_PATH}.train_regression_model") as mock_train,
            patch(f"{CALIB_CLI_PATH}.save_model"),
        ):
            mock_model = MagicMock()
            mock_train.return_value = (
                mock_model,
                {"rmse": 25.5, "mae": 20.3, "r": 0.912},
            )

            result: Result = runner.invoke(
                calibration_group,
                ["fit", str(dataset_path), "--out", str(output_path)],
            )

        assert result.exit_code == 0
        assert "RMSE: 25.5L" in result.output
        assert "MAE: 20.3L" in result.output
        assert "r: 0.912" in result.output


class TestReadCatalog:
    """Tests for _read_catalog helper function."""

    def test_read_catalog_returns_list_of_dicts(self, tmp_path: Path) -> None:
        """Test that _read_catalog returns a list of dictionaries."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title\n"
            "text1,350L,Title 1\n"
            "text2,400L,Title 2\n",
            encoding="utf-8",
        )

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _read_catalog,  # pyright: ignore[reportPrivateUsage]
        )

        result = _read_catalog(catalog)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["text_id"] == "text1"
        assert result[0]["lexile_official"] == "350L"
        assert result[1]["text_id"] == "text2"

    def test_read_catalog_empty_file(self, tmp_path: Path) -> None:
        """Test _read_catalog with empty catalog (headers only)."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text("text_id,lexile_official\n", encoding="utf-8")

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _read_catalog,  # pyright: ignore[reportPrivateUsage]
        )

        result = _read_catalog(catalog)

        assert result == []

    def test_read_catalog_handles_utf8(self, tmp_path: Path) -> None:
        """Test _read_catalog with UTF-8 content."""
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "text_id,lexile_official,title\n" "text1,350L,Títle Öne\n",
            encoding="utf-8",
        )

        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _read_catalog,  # pyright: ignore[reportPrivateUsage]
        )

        result = _read_catalog(catalog)

        assert result[0]["title"] == "Títle Öne"


class TestParseLexileValue:
    """Tests for _parse_lexile_value helper function."""

    def test_parse_lexile_value_simple_number(self) -> None:
        """Test parsing simple numeric lexile value."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("350")
        assert result == 350.0

    def test_parse_lexile_value_with_l_suffix(self) -> None:
        """Test parsing lexile value with L suffix."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("350L")
        assert result == 350.0

    def test_parse_lexile_value_with_lowercase_l(self) -> None:
        """Test parsing lexile value with lowercase l suffix."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("350l")
        assert result == 350.0

    def test_parse_lexile_value_with_whitespace(self) -> None:
        """Test parsing lexile value with whitespace."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("  350L  ")
        assert result == 350.0

    def test_parse_lexile_value_br_prefix(self) -> None:
        """Test parsing BR (beginning reader) lexile value."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("BR100L")
        assert result == -100.0

    def test_parse_lexile_value_br_only(self) -> None:
        """Test parsing BR prefix without number."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("BR")
        assert result == 0.0

    def test_parse_lexile_value_none_input(self) -> None:
        """Test parsing None input."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value(None)
        assert result is None

    def test_parse_lexile_value_empty_string(self) -> None:
        """Test parsing empty string."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("")
        assert result is None

    def test_parse_lexile_value_whitespace_only(self) -> None:
        """Test parsing whitespace-only string."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("   ")
        assert result is None

    def test_parse_lexile_value_invalid_string(self) -> None:
        """Test parsing invalid string."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _parse_lexile_value,  # pyright: ignore[reportPrivateUsage]
        )

        result = _parse_lexile_value("invalid")
        assert result is None


class TestFetchGutenbergText:
    """Tests for _fetch_gutenberg_text helper function."""

    def test_fetch_gutenberg_text_success(self, tmp_path: Path) -> None:
        """Test successful Gutenberg text fetch."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_gutenberg_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "ebook.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "Gutenberg text content"
            mock_get.return_value = mock_response

            _fetch_gutenberg_text("12345", dest)

        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == "Gutenberg text content"

    def test_fetch_gutenberg_text_tries_multiple_urls(self, tmp_path: Path) -> None:
        """Test that Gutenberg fetch tries multiple URL patterns."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_gutenberg_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "ebook.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            # First URL fails, second succeeds
            mock_fail = MagicMock()
            mock_fail.status_code = 404

            mock_success = MagicMock()
            mock_success.status_code = 200
            mock_success.text = "Success text"

            mock_get.side_effect = [mock_fail, mock_success]

            _fetch_gutenberg_text("12345", dest)

        assert dest.exists()
        assert mock_get.call_count == 2

    def test_fetch_gutenberg_text_all_urls_fail(self, tmp_path: Path) -> None:
        """Test that RuntimeError is raised when all URLs fail."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_gutenberg_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "ebook.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_fail = MagicMock()
            mock_fail.status_code = 404
            mock_get.return_value = mock_fail

            with pytest.raises(RuntimeError, match="Could not download"):
                _fetch_gutenberg_text("12345", dest)

    def test_fetch_gutenberg_text_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_gutenberg_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "nested" / "dir" / "ebook.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "Text content"
            mock_get.return_value = mock_response

            _fetch_gutenberg_text("12345", dest)

        assert dest.exists()


class TestFetchHttpText:
    """Tests for _fetch_http_text helper function."""

    def test_fetch_http_text_success(self, tmp_path: Path) -> None:
        """Test successful HTTP text fetch."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_http_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "text.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "Plain text content"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            _fetch_http_text("http://example.com/text.txt", dest)

        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == "Plain text content"

    def test_fetch_http_text_empty_url_raises_error(self, tmp_path: Path) -> None:
        """Test that empty URL raises ValueError."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_http_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "text.txt"

        with pytest.raises(ValueError, match="requires a URL"):
            _fetch_http_text("", dest)

    def test_fetch_http_text_strips_html(self, tmp_path: Path) -> None:
        """Test that HTML content is stripped."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_http_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "text.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "<html><body><p>Paragraph text</p></body></html>"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            _fetch_http_text("http://example.com/page.html", dest)

        content = dest.read_text(encoding="utf-8")
        assert "<html>" not in content
        assert "paragraph text" in content.lower()

    def test_fetch_http_text_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _fetch_http_text,  # pyright: ignore[reportPrivateUsage]
        )

        dest = tmp_path / "nested" / "dir" / "text.txt"

        with patch(f"{CALIB_CLI_PATH}.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "Text content"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            _fetch_http_text("http://example.com/text.txt", dest)

        assert dest.exists()


class TestStripHtml:
    """Tests for _strip_html helper function."""

    def test_strip_html_basic(self) -> None:
        """Test basic HTML stripping."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _strip_html,  # pyright: ignore[reportPrivateUsage]
        )

        html = "<html><body><p>Hello World</p></body></html>"
        result = _strip_html(html)

        assert "hello world" in result.lower()
        assert "<" not in result

    def test_strip_html_with_nested_tags(self) -> None:
        """Test stripping nested HTML tags."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _strip_html,  # pyright: ignore[reportPrivateUsage]
        )

        html = "<div><span><b>Bold text</b></span></div>"
        result = _strip_html(html)

        assert "bold text" in result.lower()
        assert "<" not in result

    def test_strip_html_preserves_text_content(self) -> None:
        """Test that text content is preserved."""
        from lexile_corpus_tuner.lexile_scoring_model.calibration.cli import (
            _strip_html,  # pyright: ignore[reportPrivateUsage]
        )

        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = _strip_html(html)

        assert "first paragraph" in result.lower()
        assert "second paragraph" in result.lower()

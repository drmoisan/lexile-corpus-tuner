"""Tests for corpus CLI module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result
from lexile_corpus_tuner.lexile_scoring_model.corpus.cli import corpus_group
from pytest import MonkeyPatch

CORPUS_CLI_PATH = "lexile_corpus_tuner.lexile_scoring_model.corpus"


class TestCorpusCLI:
    """Tests for corpus CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_corpus_download(self, runner: CliRunner, monkeypatch: MonkeyPatch):
        """Test corpus download command."""
        mock_ensure_dirs = MagicMock()
        mock_download_gutenberg = MagicMock()
        mock_download_wiki = MagicMock()
        mock_download_oer = MagicMock()

        monkeypatch.setattr(f"{CORPUS_CLI_PATH}.download.ensure_dirs", mock_ensure_dirs)
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_gutenberg_subset",
            mock_download_gutenberg,
        )
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_simple_wiki_dump",
            mock_download_wiki,
        )
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_oer_sources",
            mock_download_oer,
        )

        result: Result = runner.invoke(
            corpus_group, ["download", "--gutenberg-limit", "10"]
        )

        assert result.exit_code == 0
        mock_ensure_dirs.assert_called_once()
        mock_download_gutenberg.assert_called_once_with(limit=10)
        mock_download_wiki.assert_called_once()
        mock_download_oer.assert_called_once()

    def test_corpus_normalize(self, runner: CliRunner, monkeypatch: MonkeyPatch):
        """Test corpus normalize command."""
        mock_normalize = MagicMock()
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.normalize.normalize_all_sources",
            mock_normalize,
        )

        result: Result = runner.invoke(
            corpus_group, ["normalize", "--shard-size-tokens", "5000"]
        )

        assert result.exit_code == 0
        mock_normalize.assert_called_once_with(
            shard_size_tokens=5000, allowed_sources=None
        )

    def test_corpus_frequencies(self, runner: CliRunner, monkeypatch: MonkeyPatch):
        """Test corpus frequencies command."""
        mock_compute = MagicMock()
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.frequencies.compute_global_frequencies",
            mock_compute,
        )

        result: Result = runner.invoke(corpus_group, ["frequencies"])

        assert result.exit_code == 0
        mock_compute.assert_called_once_with(weighted=False, config_path=None)

    def test_corpus_frequencies_weighted_config(
        self, runner: CliRunner, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test corpus frequencies weighted flag with config path."""
        mock_compute = MagicMock()
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.frequencies.compute_global_frequencies",
            mock_compute,
        )

        config_path = tmp_path / "weights.yaml"
        result: Result = runner.invoke(
            corpus_group,
            ["frequencies", "--weighted", "--config", str(config_path)],
        )

        assert result.exit_code == 0
        mock_compute.assert_called_once_with(weighted=True, config_path=config_path)

    def test_corpus_download_filters_sources(
        self, runner: CliRunner, monkeypatch: MonkeyPatch
    ) -> None:
        """Test download command honors source selection."""
        mock_ensure_dirs = MagicMock()
        mock_download_gutenberg = MagicMock()
        mock_download_wiki = MagicMock()
        mock_download_oer = MagicMock()

        monkeypatch.setattr(f"{CORPUS_CLI_PATH}.download.ensure_dirs", mock_ensure_dirs)
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_gutenberg_subset",
            mock_download_gutenberg,
        )
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_simple_wiki_dump",
            mock_download_wiki,
        )
        monkeypatch.setattr(
            f"{CORPUS_CLI_PATH}.download.download_oer_sources",
            mock_download_oer,
        )

        result: Result = runner.invoke(
            corpus_group,
            ["download", "--gutenberg-limit", "5", "--sources", "gutenberg,oer"],
        )

        assert result.exit_code == 0
        mock_ensure_dirs.assert_called_once()
        mock_download_gutenberg.assert_called_once_with(limit=5)
        mock_download_oer.assert_called_once()
        mock_download_wiki.assert_not_called()

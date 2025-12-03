"""Tests for corpus CLI module."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result
from pytest import MonkeyPatch

from lexile_corpus_tuner.corpus.cli import corpus_group


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

        monkeypatch.setattr(
            "lexile_corpus_tuner.corpus.download.ensure_dirs", mock_ensure_dirs
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.corpus.download.download_gutenberg_subset",
            mock_download_gutenberg,
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.corpus.download.download_simple_wiki_dump",
            mock_download_wiki,
        )
        monkeypatch.setattr(
            "lexile_corpus_tuner.corpus.download.download_oer_sources",
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
            "lexile_corpus_tuner.corpus.normalize.normalize_all_sources",
            mock_normalize,
        )

        result: Result = runner.invoke(
            corpus_group, ["normalize", "--shard-size-tokens", "5000"]
        )

        assert result.exit_code == 0
        mock_normalize.assert_called_once_with(shard_size_tokens=5000)

    def test_corpus_frequencies(self, runner: CliRunner, monkeypatch: MonkeyPatch):
        """Test corpus frequencies command."""
        mock_compute = MagicMock()
        monkeypatch.setattr(
            "lexile_corpus_tuner.corpus.frequencies.compute_global_frequencies",
            mock_compute,
        )

        result: Result = runner.invoke(corpus_group, ["frequencies"])

        assert result.exit_code == 0
        mock_compute.assert_called_once()

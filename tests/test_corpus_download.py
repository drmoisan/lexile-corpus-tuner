"""Tests for corpus/download.py module.

Comprehensive tests for the download functions following unit-test-policy.md.
All network calls are mocked to ensure tests run without internet access.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from lexile_corpus_tuner.lexile_scoring_model.corpus import download

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class TestEnsureDirs:
    """Tests for ensure_dirs function."""

    def test_creates_all_required_directories(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that ensure_dirs creates all expected directories."""
        # Arrange
        fake_raw_root = tmp_path / "data" / "corpus" / "raw"
        monkeypatch.setattr(download, "RAW_ROOT", fake_raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", fake_raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", fake_raw_root / "simple_wiki")

        # Act
        download.ensure_dirs()

        # Assert
        assert (fake_raw_root / "gutenberg").exists()
        assert (fake_raw_root / "simple_wiki").exists()
        assert (fake_raw_root / "openstax").exists()
        assert (fake_raw_root / "ck12").exists()

    def test_idempotent_when_dirs_exist(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that ensure_dirs does not error if directories already exist."""
        # Arrange
        fake_raw_root = tmp_path / "data" / "corpus" / "raw"
        (fake_raw_root / "gutenberg").mkdir(parents=True)
        monkeypatch.setattr(download, "RAW_ROOT", fake_raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", fake_raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", fake_raw_root / "simple_wiki")

        # Act / Assert - should not raise
        download.ensure_dirs()
        assert (fake_raw_root / "gutenberg").exists()


class TestIterGutenbergIds:
    """Tests for _iter_gutenberg_ids function."""

    def test_yields_ids_from_file(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that IDs are correctly parsed from the file."""
        # Arrange
        ids_file = tmp_path / "gutenberg_ids.txt"
        ids_file.write_text("1\n100\n1000\n")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(None))

        # Assert
        assert ids == [1, 100, 1000]

    def test_respects_limit(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test that limit parameter restricts number of IDs returned."""
        # Arrange
        ids_file = tmp_path / "gutenberg_ids.txt"
        ids_file.write_text("1\n2\n3\n4\n5\n")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(2))

        # Assert
        assert ids == [1, 2]

    def test_skips_comments_and_blank_lines(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that comments and blank lines are ignored."""
        # Arrange
        ids_file = tmp_path / "gutenberg_ids.txt"
        ids_file.write_text("# Header comment\n1\n\n# Another comment\n2\n   \n3\n")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(None))

        # Assert
        assert ids == [1, 2, 3]

    def test_skips_invalid_ids(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test that non-numeric lines are skipped."""
        # Arrange
        ids_file = tmp_path / "gutenberg_ids.txt"
        ids_file.write_text("1\ninvalid\n2\nabc123\n3\n")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(None))

        # Assert
        assert ids == [1, 2, 3]

    def test_returns_empty_if_file_missing(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that missing file yields no IDs."""
        # Arrange
        ids_file = tmp_path / "nonexistent.txt"
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(None))

        # Assert
        assert ids == []

    def test_returns_empty_for_empty_file(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that empty file yields no IDs."""
        # Arrange
        ids_file = tmp_path / "empty.txt"
        ids_file.write_text("")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act
        iter_fn = download._iter_gutenberg_ids  # pyright: ignore[reportPrivateUsage]
        ids = list(iter_fn(None))

        # Assert
        assert ids == []


class TestResolveGutenbergUrl:
    """Tests for _resolve_gutenberg_url function."""

    @patch("requests.head")
    def test_returns_first_successful_url(self, mock_head: Mock) -> None:
        """Test that the first URL with 200 status is returned."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Act
        resolve_fn = (
            download._resolve_gutenberg_url  # pyright: ignore[reportPrivateUsage]
        )
        result = resolve_fn(123)

        # Assert
        assert result is not None
        assert "123" in result
        mock_head.assert_called_once()

    @patch("requests.head")
    def test_tries_multiple_patterns(self, mock_head: Mock) -> None:
        """Test that multiple URL patterns are tried when first fails."""
        # Arrange - First call returns 404, second returns 200
        mock_404 = Mock(status_code=404)
        mock_200 = Mock(status_code=200)
        mock_head.side_effect = [mock_404, mock_200]

        # Act
        resolve_fn = (
            download._resolve_gutenberg_url  # pyright: ignore[reportPrivateUsage]
        )
        result = resolve_fn(456)

        # Assert
        assert result is not None
        assert mock_head.call_count == 2

    @patch("requests.head")
    def test_returns_none_when_all_fail(self, mock_head: Mock) -> None:
        """Test that None is returned when no URL pattern succeeds."""
        # Arrange
        mock_response = Mock(status_code=404)
        mock_head.return_value = mock_response

        # Act
        resolve_fn = (
            download._resolve_gutenberg_url  # pyright: ignore[reportPrivateUsage]
        )
        result = resolve_fn(789)

        # Assert
        assert result is None
        # Should have tried all 3 patterns
        assert mock_head.call_count == 3

    @patch("requests.head")
    def test_handles_network_errors(self, mock_head: Mock) -> None:
        """Test that network errors are caught and next URL is tried."""
        # Arrange - First call raises exception, second succeeds
        mock_head.side_effect = [
            requests.RequestException("Network error"),
            Mock(status_code=200),
        ]

        # Act
        resolve_fn = (
            download._resolve_gutenberg_url  # pyright: ignore[reportPrivateUsage]
        )
        result = resolve_fn(999)

        # Assert
        assert result is not None
        assert mock_head.call_count == 2

    @patch("requests.head")
    def test_returns_none_when_all_network_errors(self, mock_head: Mock) -> None:
        """Test that None is returned when all requests fail with network errors."""
        # Arrange
        mock_head.side_effect = requests.RequestException("Network error")

        # Act
        resolve_fn = (
            download._resolve_gutenberg_url  # pyright: ignore[reportPrivateUsage]
        )
        result = resolve_fn(111)

        # Assert
        assert result is None
        assert mock_head.call_count == 3


class TestDownloadFile:
    """Tests for _download_file function."""

    @patch("requests.get")
    def test_downloads_file_successfully(self, mock_get: Mock, tmp_path: Path) -> None:
        """Test that file is downloaded and written correctly."""
        # Arrange
        dest = tmp_path / "output" / "test.txt"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        # Act
        download_fn = download._download_file  # pyright: ignore[reportPrivateUsage]
        download_fn("http://example.com/file.txt", dest)

        # Assert
        assert dest.exists()
        assert dest.read_bytes() == b"chunk1chunk2"

    @patch("requests.get")
    def test_creates_parent_directories(self, mock_get: Mock, tmp_path: Path) -> None:
        """Test that parent directories are created if they don't exist."""
        # Arrange
        dest = tmp_path / "deep" / "nested" / "path" / "file.txt"
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"test"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        # Act
        download_fn = download._download_file  # pyright: ignore[reportPrivateUsage]
        download_fn("http://example.com/file.txt", dest)

        # Assert
        assert dest.exists()

    @patch("requests.get")
    def test_skips_empty_chunks(self, mock_get: Mock, tmp_path: Path) -> None:
        """Test that empty chunks are skipped during download."""
        # Arrange
        dest = tmp_path / "test.txt"
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"a", b"", b"b", b"", b"c"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        # Act
        download_fn = download._download_file  # pyright: ignore[reportPrivateUsage]
        download_fn("http://example.com/file.txt", dest)

        # Assert
        assert dest.read_bytes() == b"abc"

    @patch("requests.get")
    def test_raises_on_http_error(self, mock_get: Mock, tmp_path: Path) -> None:
        """Test that HTTP errors are propagated."""
        # Arrange
        dest = tmp_path / "test.txt"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        # Act / Assert
        download_fn = download._download_file  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(requests.HTTPError):
            download_fn("http://example.com/file.txt", dest)


class TestCopyLocalFile:
    """Tests for _copy_local_file function."""

    def test_copies_file_successfully(self, tmp_path: Path) -> None:
        """Test that local file is copied correctly."""
        # Arrange
        src = tmp_path / "source.txt"
        dest = tmp_path / "dest" / "output.txt"
        src.write_text("Hello World")

        # Act
        copy_fn = download._copy_local_file  # pyright: ignore[reportPrivateUsage]
        copy_fn(src, dest)

        # Assert
        assert dest.exists()
        assert dest.read_text() == "Hello World"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if they don't exist."""
        # Arrange
        src = tmp_path / "source.txt"
        dest = tmp_path / "deep" / "nested" / "dest.txt"
        src.write_text("content")

        # Act
        copy_fn = download._copy_local_file  # pyright: ignore[reportPrivateUsage]
        copy_fn(src, dest)

        # Assert
        assert dest.exists()

    def test_raises_on_missing_source(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing source."""
        # Arrange
        src = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        # Act / Assert
        copy_fn = download._copy_local_file  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(FileNotFoundError) as exc_info:
            copy_fn(src, dest)
        assert "Source file not found" in str(exc_info.value)


class TestDownloadGutenbergSubset:
    """Tests for download_gutenberg_subset function."""

    def test_skips_already_downloaded_files(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that existing files are skipped."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        gutenberg_dir = raw_root / "gutenberg"
        gutenberg_dir.mkdir(parents=True)
        (gutenberg_dir / "1.txt").write_text("existing")

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1\n")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", gutenberg_dir)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        mock_resolve = MagicMock(return_value="http://example.com/1.txt")
        mock_download = MagicMock()
        monkeypatch.setattr(download, "_resolve_gutenberg_url", mock_resolve)
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        download.download_gutenberg_subset()

        # Assert - download should not be called for existing file
        mock_download.assert_not_called()

    def test_warns_when_no_ids_file(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warning is logged when IDs file is missing."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        ids_file = tmp_path / "nonexistent.txt"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        # Act

        with caplog.at_level(logging.WARNING):
            download.download_gutenberg_subset()

        # Assert
        assert "No Gutenberg IDs found" in caplog.text

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._resolve_gutenberg_url"
    )
    def test_downloads_new_files(
        self,
        mock_resolve: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test that new files are downloaded successfully."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        gutenberg_dir = raw_root / "gutenberg"
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1\n2\n")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", gutenberg_dir)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        mock_resolve.return_value = "http://example.com/file.txt"

        # Act
        download.download_gutenberg_subset()

        # Assert
        assert mock_resolve.call_count == 2
        assert mock_download.call_count == 2

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._resolve_gutenberg_url"
    )
    def test_warns_when_url_resolution_fails(
        self,
        mock_resolve: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warning is logged when URL resolution fails."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        gutenberg_dir = raw_root / "gutenberg"
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("999\n")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", gutenberg_dir)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        mock_resolve.return_value = None

        # Act

        with caplog.at_level(logging.WARNING):
            download.download_gutenberg_subset()

        # Assert
        mock_download.assert_not_called()
        assert "Unable to construct URL" in caplog.text

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._resolve_gutenberg_url"
    )
    def test_continues_on_download_error(
        self,
        mock_resolve: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that download continues for other files when one fails."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        gutenberg_dir = raw_root / "gutenberg"
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1\n2\n")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", gutenberg_dir)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        mock_resolve.return_value = "http://example.com/file.txt"
        mock_download.side_effect = [
            requests.RequestException("Network error"),
            None,  # Second download succeeds
        ]

        # Act

        with caplog.at_level(logging.ERROR):
            download.download_gutenberg_subset()

        # Assert
        assert mock_download.call_count == 2
        assert "Failed to download Gutenberg 1" in caplog.text

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._resolve_gutenberg_url"
    )
    def test_respects_limit_parameter(
        self,
        mock_resolve: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test that limit parameter restricts number of downloads."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        gutenberg_dir = raw_root / "gutenberg"
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1\n2\n3\n4\n5\n")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", gutenberg_dir)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "GUTENBERG_IDS_FILE", ids_file)

        mock_resolve.return_value = "http://example.com/file.txt"

        # Act
        download.download_gutenberg_subset(limit=2)

        # Assert
        assert mock_download.call_count == 2


class TestDownloadSimpleWikiDump:
    """Tests for download_simple_wiki_dump function."""

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_downloads_from_provided_url(
        self, mock_download: Mock, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that dump is downloaded from provided URL."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        wiki_dir = raw_root / "simple_wiki"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", wiki_dir)

        # Act
        result = download.download_simple_wiki_dump(
            "http://example.com/wiki_dump.xml.bz2"
        )

        # Assert
        mock_download.assert_called_once()
        assert result == wiki_dir / "wiki_dump.xml.bz2"

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_uses_default_url_when_none_provided(
        self, mock_download: Mock, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that default URL is used when none is provided."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        wiki_dir = raw_root / "simple_wiki"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", wiki_dir)
        monkeypatch.delenv("LEXILE_SIMPLE_WIKI_DUMP_URL", raising=False)

        # Act
        download.download_simple_wiki_dump()

        # Assert
        mock_download.assert_called_once()
        call_url: str = mock_download.call_args[0][0]
        assert "wikimedia.org" in call_url

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_uses_environment_variable_url(
        self, mock_download: Mock, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that environment variable URL is used when set."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        wiki_dir = raw_root / "simple_wiki"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", wiki_dir)
        monkeypatch.setenv("LEXILE_SIMPLE_WIKI_DUMP_URL", "http://custom.url/dump.bz2")

        # Act
        download.download_simple_wiki_dump()

        # Assert
        mock_download.assert_called_once()
        call_url: str = mock_download.call_args[0][0]
        assert call_url == "http://custom.url/dump.bz2"

    def test_skips_download_if_exists(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that download is skipped if file already exists."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        wiki_dir = raw_root / "simple_wiki"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "existing.xml.bz2").write_text("existing content")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", wiki_dir)

        mock_download = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        with caplog.at_level(logging.INFO):
            result = download.download_simple_wiki_dump(
                "http://example.com/existing.xml.bz2"
            )

        # Assert
        mock_download.assert_not_called()
        assert "already exists" in caplog.text
        assert result == wiki_dir / "existing.xml.bz2"

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_handles_url_without_filename(
        self, mock_download: Mock, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that fallback filename is used for URLs without filename."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        wiki_dir = raw_root / "simple_wiki"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", wiki_dir)

        # Act
        result = download.download_simple_wiki_dump("http://example.com/")

        # Assert
        assert result.name == "simplewiki_dump.xml.bz2"


class TestDownloadOerSources:
    """Tests for download_oer_sources function."""

    def test_skips_when_manifest_missing(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that download is skipped when manifest file is missing."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "nonexistent_manifest.json"

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        mock_download = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        with caplog.at_level(logging.INFO):
            download.download_oer_sources()

        # Assert
        mock_download.assert_not_called()
        assert "OER manifest missing" in caplog.text

    def test_handles_invalid_json(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid JSON is handled gracefully."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text("invalid json {{{")

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        with caplog.at_level(logging.ERROR):
            download.download_oer_sources()

        # Assert
        assert "Failed to parse" in caplog.text

    def test_skips_when_no_sources(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that download is skipped when sources list is empty."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "manifest.json"
        manifest_content: dict[str, list[Any]] = {"sources": []}
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        mock_download = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        with caplog.at_level(logging.INFO):
            download.download_oer_sources()

        # Assert
        mock_download.assert_not_called()
        assert "No sources listed" in caplog.text

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_downloads_from_http_url(
        self, mock_download: Mock, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that HTTP URLs are downloaded correctly."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "book1",
                    "url": "http://example.com/book1.txt",
                    "source_id": "openstax",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        download.download_oer_sources()

        # Assert
        mock_download.assert_called_once()
        call_url, call_dest = mock_download.call_args[0]
        assert call_url == "http://example.com/book1.txt"
        assert "openstax" in str(call_dest)

    def test_copies_local_file_with_file_url(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that file:// URLs trigger local copy."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        source_file = tmp_path / "source.txt"
        source_file.write_text("content from local file")

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "local1",
                    "url": f"file://{source_file}",
                    "source_id": "openstax",
                    "filename": "local1.txt",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        download.download_oer_sources()

        # Assert
        dest_file = raw_root / "openstax" / "local1.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "content from local file"

    def test_copies_local_file_with_absolute_path(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that absolute paths (starting with /) trigger local copy."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        source_file = tmp_path / "source.txt"
        source_file.write_text("content from absolute path")

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "abs1",
                    "url": str(source_file),  # Absolute path
                    "source_id": "ck12",
                    "filename": "abs1.txt",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        download.download_oer_sources()

        # Assert
        dest_file = raw_root / "ck12" / "abs1.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "content from absolute path"

    def test_skips_entries_without_url(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that entries without URL are skipped with warning."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [{"id": "nourl1", "source_id": "openstax"}]  # No URL
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        mock_download = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        with caplog.at_level(logging.WARNING):
            download.download_oer_sources()

        # Assert
        mock_download.assert_not_called()
        assert "missing URL" in caplog.text

    def test_skips_already_downloaded_files(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that existing files are skipped."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        openstax_dir = raw_root / "openstax"
        openstax_dir.mkdir(parents=True)
        (openstax_dir / "existing.txt").write_text("existing")

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "book1",
                    "url": "http://example.com/book.txt",
                    "source_id": "openstax",
                    "filename": "existing.txt",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        mock_download = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)

        # Act
        with caplog.at_level(logging.INFO):
            download.download_oer_sources()

        # Assert
        mock_download.assert_not_called()
        assert "already downloaded" in caplog.text

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    def test_continues_on_download_error(
        self,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that download continues for other sources when one fails."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "book1",
                    "url": "http://example.com/book1.txt",
                    "source_id": "openstax",
                },
                {
                    "id": "book2",
                    "url": "http://example.com/book2.txt",
                    "source_id": "openstax",
                },
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        mock_download.side_effect = [
            requests.RequestException("Network error"),
            None,  # Second download succeeds
        ]

        # Act
        with caplog.at_level(logging.ERROR):
            download.download_oer_sources()

        # Assert
        assert mock_download.call_count == 2
        assert "Failed to download" in caplog.text

    def test_uses_default_source_id(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that default source_id 'oer' is used when not provided."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        source_file = tmp_path / "source.txt"
        source_file.write_text("content")

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "book1",
                    "url": str(source_file),
                    # No source_id provided
                    "filename": "book1.txt",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        download.download_oer_sources()

        # Assert - should use "oer" as default source_id
        dest_file = raw_root / "oer" / "book1.txt"
        assert dest_file.exists()

    def test_uses_item_id_as_filename_fallback(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that item ID is used as filename when filename not provided."""
        # Arrange
        raw_root = tmp_path / "data" / "corpus" / "raw"
        source_file = tmp_path / "source.txt"
        source_file.write_text("content")

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "my_book",
                    "url": str(source_file),
                    "source_id": "openstax",
                    # No filename provided
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        download.download_oer_sources()

        # Assert - should use "{item_id}.txt" as filename
        dest_file = raw_root / "openstax" / "my_book.txt"
        assert dest_file.exists()

    def test_handles_copy_error(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that copy errors are handled gracefully."""
        # Arrange

        raw_root = tmp_path / "data" / "corpus" / "raw"
        nonexistent_source = tmp_path / "nonexistent.txt"

        manifest_file = tmp_path / "manifest.json"
        manifest_content = {
            "sources": [
                {
                    "id": "book1",
                    "url": str(nonexistent_source),
                    "source_id": "openstax",
                    "filename": "book1.txt",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_content))

        monkeypatch.setattr(download, "RAW_ROOT", raw_root)
        monkeypatch.setattr(download, "GUTENBERG_DIR", raw_root / "gutenberg")
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", raw_root / "simple_wiki")
        monkeypatch.setattr(download, "OER_MANIFEST", manifest_file)

        # Act
        with caplog.at_level(logging.ERROR):
            download.download_oer_sources()

        # Assert
        assert "Failed to download" in caplog.text

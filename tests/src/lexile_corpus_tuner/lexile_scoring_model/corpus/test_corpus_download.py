"""Tests for corpus/download.py module.

Comprehensive tests for the download functions following unit-test-policy.md.
All network calls are mocked to ensure tests run without internet access.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from lexile_corpus_tuner.lexile_scoring_model.corpus import download

if TYPE_CHECKING:
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
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._extract_simple_wiki_bz2"
    )
    def test_downloads_from_provided_url(
        self,
        mock_extract: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
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
        mock_extract.assert_called_once()
        assert result == wiki_dir / "wiki_dump.xml.bz2"

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._extract_simple_wiki_bz2"
    )
    def test_uses_default_url_when_none_provided(
        self,
        mock_extract: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
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
        mock_extract.assert_called_once()
        call_url: str = mock_download.call_args[0][0]
        assert "wikimedia.org" in call_url

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._extract_simple_wiki_bz2"
    )
    def test_uses_environment_variable_url(
        self,
        mock_extract: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
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
        mock_extract.assert_called_once()
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
        mock_extract = MagicMock()
        monkeypatch.setattr(download, "_download_file", mock_download)
        monkeypatch.setattr(download, "_extract_simple_wiki_bz2", mock_extract)

        # Act
        with caplog.at_level(logging.INFO):
            result = download.download_simple_wiki_dump(
                "http://example.com/existing.xml.bz2"
            )

        # Assert
        mock_download.assert_not_called()
        mock_extract.assert_called_once_with(wiki_dir / "existing.xml.bz2")
        assert "already exists" in caplog.text
        assert result == wiki_dir / "existing.xml.bz2"

    @patch("lexile_corpus_tuner.lexile_scoring_model.corpus.download._download_file")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.corpus.download._extract_simple_wiki_bz2"
    )
    def test_handles_url_without_filename(
        self,
        mock_extract: Mock,
        mock_download: Mock,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
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
        mock_download.assert_called_once()
        mock_extract.assert_called_once()
        assert result.name == "simplewiki_dump.xml.bz2"


class NonClosingBytesIO(io.BytesIO):
    """
    BytesIO variant that keeps the buffer accessible after context-manager exits.

    Purpose:
        Provide an in-memory stream that behaves like a file handle but does not
        close so tests can inspect written bytes after `with` blocks.

    Usage:
        Use this in tests that need to read back data written by the code under
        test without touching the filesystem.

    Flow:
        Return this object from monkeypatched Path.open calls and inspect
        `.getvalue()` after the helper completes.

    Invariants / Constraints:
        Closing is intentionally suppressed; callers must not rely on close
        side-effects for cleanup.

    Side Effects:
        None. This is an in-memory test helper.
    """

    def close(self) -> None:
        """
        Prevent closing so tests can inspect the buffer after writes.

        Purpose:
            Override the default close behavior to keep the in-memory buffer
            readable after context-manager exits.

        Args:
            None.

        Returns:
            None: This method intentionally does not close the buffer.

        Raises:
            None.

        Side Effects:
            Leaves the buffer open so `.getvalue()` remains available.
        """


class TestExtractSimpleWikiBz2:
    """Tests for _extract_simple_wiki_bz2 helper."""

    def test_extract_simple_wiki_bz2_writes_xml_and_replaces_temp(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Verify the helper streams content and atomically replaces the XML.

        Purpose:
            Ensure `_extract_simple_wiki_bz2` writes the decompressed payload to
            a temp file and replaces it with the final XML path.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for stubbing
                file operations.

        Returns:
            None.

        Raises:
            AssertionError: If the helper does not copy data or replace paths.

        Side Effects:
            Uses in-memory streams to simulate file I/O.
        """
        # Arrange
        bz2_path = Path(
            "data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2"
        )
        xml_path = bz2_path.with_suffix(".xml")
        tmp_path = xml_path.with_suffix(".xml.tmp")
        source_stream = io.BytesIO(b"<mediawiki>payload</mediawiki>")
        temp_stream = NonClosingBytesIO()
        replaced_paths: list[tuple[Path, Path]] = []

        def fake_bz2_open(path: Path, mode: str) -> io.BytesIO:
            """
            Return a reusable in-memory source stream for the bz2 payload.

            Purpose:
                Provide a deterministic source stream for the extraction helper.

            Args:
                path (Path): Expected bz2 path for the dump.
                mode (str): Open mode; must be binary read.

            Returns:
                io.BytesIO: The in-memory stream containing the bz2 payload.

            Raises:
                AssertionError: If path or mode are unexpected.

            Side Effects:
                Resets the stream position to the beginning.
            """
            assert path == bz2_path
            assert mode == "rb"
            source_stream.seek(0)
            return source_stream

        def fake_open(
            path: Path, mode: str = "r", *args: Any, **kwargs: Any
        ) -> NonClosingBytesIO:
            """
            Return an in-memory buffer for the temporary XML output.

            Purpose:
                Replace filesystem writes with a controllable buffer.

            Args:
                path (Path): Expected temp XML path.
                mode (str): Open mode; must be binary write.
                *args (Any): Unused positional arguments from Path.open.
                **kwargs (Any): Unused keyword arguments from Path.open.

            Returns:
                NonClosingBytesIO: The buffer that captures written bytes.

            Raises:
                AssertionError: If path or mode are unexpected.

            Side Effects:
                Clears any previous contents from the buffer.
            """
            assert path == tmp_path
            assert mode == "wb"
            temp_stream.seek(0)
            temp_stream.truncate(0)
            return temp_stream

        def fake_replace(path: Path, target: Path) -> Path:
            """
            Record the atomic replace call for later assertions.

            Purpose:
                Capture the temp and final XML paths used for replacement.

            Args:
                path (Path): The temp path being replaced.
                target (Path): The final XML path.

            Returns:
                Path: The target path to mimic Path.replace behavior.

            Raises:
                None.

            Side Effects:
                Appends the replacement tuple to `replaced_paths`.
            """
            replaced_paths.append((path, target))
            return target

        monkeypatch.setattr(download.bz2, "open", fake_bz2_open)
        monkeypatch.setattr(Path, "open", fake_open)
        monkeypatch.setattr(Path, "replace", fake_replace)

        # Act
        extract_fn = (
            download._extract_simple_wiki_bz2  # pyright: ignore[reportPrivateUsage]
        )
        result = extract_fn(bz2_path)

        # Assert
        assert result == xml_path
        assert temp_stream.getvalue() == b"<mediawiki>payload</mediawiki>"
        assert replaced_paths == [(tmp_path, xml_path)]

    def test_extract_simple_wiki_bz2_cleans_temp_on_error(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Verify the helper removes temp output when extraction fails.

        Purpose:
            Ensure `_extract_simple_wiki_bz2` deletes the temporary XML file
            when a copy error occurs and re-raises the exception.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for stubbing
                file operations and failure injection.

        Returns:
            None.

        Raises:
            AssertionError: If cleanup or error propagation fails.

        Side Effects:
            Uses in-memory streams to simulate file I/O.
        """
        # Arrange
        bz2_path = Path(
            "data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2"
        )
        xml_path = bz2_path.with_suffix(".xml")
        tmp_path = xml_path.with_suffix(".xml.tmp")
        source_stream = io.BytesIO(b"broken payload")
        temp_stream = NonClosingBytesIO()
        deleted_paths: list[Path] = []
        replaced_paths: list[Path] = []

        def fake_bz2_open(path: Path, mode: str) -> io.BytesIO:
            """
            Return a deterministic in-memory source stream for extraction.

            Purpose:
                Provide a predictable input stream for the failure scenario.

            Args:
                path (Path): Expected bz2 path for the dump.
                mode (str): Open mode; must be binary read.

            Returns:
                io.BytesIO: The in-memory stream containing the bz2 payload.

            Raises:
                AssertionError: If path or mode are unexpected.

            Side Effects:
                Resets the stream position to the beginning.
            """
            assert path == bz2_path
            assert mode == "rb"
            source_stream.seek(0)
            return source_stream

        def fake_open(
            path: Path, mode: str = "r", *args: Any, **kwargs: Any
        ) -> NonClosingBytesIO:
            """
            Return an in-memory buffer for the temp XML output.

            Purpose:
                Replace filesystem writes during the failing extraction.

            Args:
                path (Path): Expected temp XML path.
                mode (str): Open mode; must be binary write.
                *args (Any): Unused positional arguments from Path.open.
                **kwargs (Any): Unused keyword arguments from Path.open.

            Returns:
                NonClosingBytesIO: The buffer that would receive output.

            Raises:
                AssertionError: If path or mode are unexpected.

            Side Effects:
                Clears any previous contents from the buffer.
            """
            assert path == tmp_path
            assert mode == "wb"
            temp_stream.seek(0)
            temp_stream.truncate(0)
            return temp_stream

        def fake_copyfileobj(
            source: io.BufferedReader, dest: io.BufferedWriter, length: int = 0
        ) -> None:
            """
            Raise an error to simulate a copy failure.

            Purpose:
                Force `_extract_simple_wiki_bz2` into its cleanup path.

            Args:
                source (io.BufferedReader): Source stream (unused).
                dest (io.BufferedWriter): Destination stream (unused).
                length (int): Chunk size requested by caller (unused).

            Returns:
                None.

            Raises:
                RuntimeError: Always raised to simulate failure.

            Side Effects:
                None.
            """
            raise RuntimeError("copy failed")

        def fake_exists(path: Path) -> bool:
            """
            Report whether the temp path exists for cleanup checks.

            Purpose:
                Ensure the helper believes the temp path exists so it attempts
                cleanup.

            Args:
                path (Path): Path to check.

            Returns:
                bool: True only for the temp path.

            Raises:
                None.

            Side Effects:
                None.
            """
            return path == tmp_path

        def fake_unlink(path: Path) -> None:
            """
            Record temp-path deletion attempts.

            Purpose:
                Capture cleanup behavior for assertions.

            Args:
                path (Path): Path scheduled for deletion.

            Returns:
                None.

            Raises:
                None.

            Side Effects:
                Adds the deleted path to `deleted_paths`.
            """
            deleted_paths.append(path)

        def fake_replace(path: Path, target: Path) -> Path:
            """
            Record replace attempts to ensure they do not occur.

            Purpose:
                Track whether the helper incorrectly performed a replace
                during a failed extraction.

            Args:
                path (Path): Temp path involved in replace.
                target (Path): Target path for replace.

            Returns:
                Path: The target path, to mimic Path.replace.

            Raises:
                None.

            Side Effects:
                Appends the temp path to `replaced_paths`.
            """
            replaced_paths.append(path)
            return target

        monkeypatch.setattr(download.bz2, "open", fake_bz2_open)
        monkeypatch.setattr(Path, "open", fake_open)
        monkeypatch.setattr(download.shutil, "copyfileobj", fake_copyfileobj)
        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "unlink", fake_unlink)
        monkeypatch.setattr(Path, "replace", fake_replace)

        # Act / Assert
        extract_fn = (
            download._extract_simple_wiki_bz2  # pyright: ignore[reportPrivateUsage]
        )
        with pytest.raises(RuntimeError, match="copy failed"):
            extract_fn(bz2_path)

        assert deleted_paths == [tmp_path]
        assert replaced_paths == []

    def test_download_simple_wiki_dump_skips_extraction_when_xml_exists(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Verify download skips extraction when XML already exists.

        Purpose:
            Confirm that a non-empty XML file causes the download flow to skip
            extraction and downloading.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for stubbing
                filesystem checks and helper functions.

        Returns:
            None.

        Raises:
            AssertionError: If extraction or download is invoked unexpectedly.

        Side Effects:
            Patches filesystem calls to avoid real disk access.
        """
        # Arrange
        bz2_path = Path(
            "data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2"
        )
        xml_path = bz2_path.with_suffix(".xml")
        extract_mock = MagicMock()
        download_mock = MagicMock()

        def fake_exists(path: Path) -> bool:
            """
            Pretend both bz2 and XML paths exist.

            Purpose:
                Provide deterministic existence checks for the skip branch.

            Args:
                path (Path): Path to check.

            Returns:
                bool: True for bz2 and XML paths.

            Raises:
                None.

            Side Effects:
                None.
            """
            return path in {bz2_path, xml_path}

        def fake_stat(path: Path) -> SimpleNamespace:
            """
            Return a non-zero size for XML to trigger the skip branch.

            Purpose:
                Simulate a valid, non-empty XML file.

            Args:
                path (Path): Path to stat.

            Returns:
                SimpleNamespace: Object with an st_size attribute.

            Raises:
                None.

            Side Effects:
                None.
            """
            if path == xml_path:
                return SimpleNamespace(st_size=100)
            return SimpleNamespace(st_size=0)

        monkeypatch.setattr(download, "ensure_dirs", lambda: None)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", bz2_path.parent)
        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "stat", fake_stat)
        monkeypatch.setattr(download, "_extract_simple_wiki_bz2", extract_mock)
        monkeypatch.setattr(download, "_download_file", download_mock)

        # Act
        download.download_simple_wiki_dump(
            "https://example.com/simplewiki-latest-pages-articles.xml.bz2"
        )

        # Assert
        extract_mock.assert_not_called()
        download_mock.assert_not_called()

    def test_download_simple_wiki_dump_extracts_when_xml_missing(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Verify extraction runs when the XML is missing.

        Purpose:
            Ensure the download flow extracts when the compressed dump exists
            but the XML output is absent.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for stubbing
                filesystem checks and helper functions.

        Returns:
            None.

        Raises:
            AssertionError: If download is invoked or extraction is skipped.

        Side Effects:
            Patches filesystem calls to avoid real disk access.
        """
        # Arrange
        bz2_path = Path(
            "data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2"
        )
        xml_path = bz2_path.with_suffix(".xml")
        extract_mock = MagicMock()
        download_mock = MagicMock()

        def fake_exists(path: Path) -> bool:
            """
            Report bz2 exists while XML does not.

            Purpose:
                Force the extraction branch without triggering download.

            Args:
                path (Path): Path to check.

            Returns:
                bool: True for bz2, False for XML and others.

            Raises:
                None.

            Side Effects:
                None.
            """
            if path == bz2_path:
                return True
            if path == xml_path:
                return False
            return False

        monkeypatch.setattr(download, "ensure_dirs", lambda: None)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", bz2_path.parent)
        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(download, "_extract_simple_wiki_bz2", extract_mock)
        monkeypatch.setattr(download, "_download_file", download_mock)

        # Act
        download.download_simple_wiki_dump(
            "https://example.com/simplewiki-latest-pages-articles.xml.bz2"
        )

        # Assert
        extract_mock.assert_called_once_with(bz2_path)
        download_mock.assert_not_called()

    def test_download_simple_wiki_dump_downloads_then_extracts(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Verify download occurs before extraction when bz2 is missing.

        Purpose:
            Confirm the flow downloads the dump and then invokes extraction
            when the bz2 file is absent.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for stubbing
                filesystem checks and helper functions.

        Returns:
            None.

        Raises:
            AssertionError: If the call order differs from expectations.

        Side Effects:
            Records call order without touching the filesystem.
        """
        # Arrange
        bz2_path = Path(
            "data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2"
        )
        call_order: list[str] = []

        def fake_exists(path: Path) -> bool:
            """
            Indicate that no files exist yet.

            Purpose:
                Force the download branch by reporting missing paths.

            Args:
                path (Path): Path to check.

            Returns:
                bool: Always False.

            Raises:
                None.

            Side Effects:
                None.
            """
            return False

        def fake_download(url: str, dest: Path) -> None:
            """
            Record the download call.

            Purpose:
                Capture the download step ordering.

            Args:
                url (str): Download URL (unused beyond logging).
                dest (Path): Destination path (unused beyond logging).

            Returns:
                None.

            Raises:
                None.

            Side Effects:
                Appends "download" to the call order list.
            """
            call_order.append("download")

        def fake_extract(path: Path) -> Path:
            """
            Record the extract call and return the XML path.

            Purpose:
                Capture the extraction step ordering.

            Args:
                path (Path): BZ2 source path for extraction.

            Returns:
                Path: Derived XML path used by the helper.

            Raises:
                None.

            Side Effects:
                Appends "extract" to the call order list.
            """
            call_order.append("extract")
            return path.with_suffix(".xml")

        monkeypatch.setattr(download, "ensure_dirs", lambda: None)
        monkeypatch.setattr(download, "SIMPLE_WIKI_DIR", bz2_path.parent)
        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(download, "_download_file", fake_download)
        monkeypatch.setattr(download, "_extract_simple_wiki_bz2", fake_extract)

        # Act
        download.download_simple_wiki_dump(
            "https://example.com/simplewiki-latest-pages-articles.xml.bz2"
        )

        # Assert
        assert call_order == ["download", "extract"]


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

"""Unit tests for scripts/production/build_gutenberg_id_list.py.

Tests the Gutenberg metadata fetcher with checkpoint/resume support.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import requests

if TYPE_CHECKING:
    from pathlib import Path

from scripts.production.build_gutenberg_id_list import (
    ParquetStore,
    fetch_books_incremental,
    load_checkpoint,
    save_checkpoint,
)


class InMemoryParquetStore(ParquetStore):
    """Simple in-memory parquet store for isolating tests from the filesystem."""

    def __init__(self) -> None:
        self._buffer: bytes | None = None

    def exists(self) -> bool:
        return self._buffer is not None

    def load(self) -> pd.DataFrame:
        if self._buffer is None:
            raise FileNotFoundError("No parquet data has been written")
        # pandas exposes parquet kwargs as Unknown in the stub; ignore for type
        # checker only.
        return pd.read_parquet(io.BytesIO(self._buffer))  # type: ignore[reportUnknownMemberType]

    def save(self, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        # pandas exposes parquet kwargs as Unknown in the stub; ignore for type
        # checker only.
        df.to_parquet(buffer, index=False)  # type: ignore[reportUnknownMemberType]
        self._buffer = buffer.getvalue()


@pytest.fixture()
def memory_parquet_store() -> InMemoryParquetStore:
    """Provide an isolated in-memory parquet store per test."""

    return InMemoryParquetStore()


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_returns_zero_when_checkpoint_missing(self, tmp_path: Path) -> None:
        """Test that missing checkpoint returns 0."""
        checkpoint_path = tmp_path / "nonexistent.json"

        result = load_checkpoint(checkpoint_path)

        assert result == 0

    def test_loads_last_page_from_checkpoint(self, tmp_path: Path) -> None:
        """Test loading last_page from valid checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_data = {"last_page": 42}
        checkpoint_path.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        result = load_checkpoint(checkpoint_path)

        assert result == 42

    def test_returns_zero_on_malformed_json(self, tmp_path: Path) -> None:
        """Test that malformed JSON returns 0."""
        checkpoint_path = tmp_path / "bad.json"
        checkpoint_path.write_text("not valid json", encoding="utf-8")

        result = load_checkpoint(checkpoint_path)

        assert result == 0

    def test_returns_zero_when_last_page_missing(self, tmp_path: Path) -> None:
        """Test that checkpoint without last_page key returns 0."""
        checkpoint_path = tmp_path / "incomplete.json"
        checkpoint_path.write_text(json.dumps({"other_key": 123}), encoding="utf-8")

        result = load_checkpoint(checkpoint_path)

        assert result == 0


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_creates_checkpoint_file(self, tmp_path: Path) -> None:
        """Test that checkpoint file is created with correct data."""
        checkpoint_path = tmp_path / "checkpoint.json"

        save_checkpoint(checkpoint_path, 15)

        assert checkpoint_path.exists()
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert data["last_page"] == 15

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if needed."""
        checkpoint_path = tmp_path / "nested" / "dir" / "checkpoint.json"

        save_checkpoint(checkpoint_path, 99)

        assert checkpoint_path.exists()
        assert checkpoint_path.parent.exists()

    def test_overwrites_existing_checkpoint(self, tmp_path: Path) -> None:
        """Test that existing checkpoint is overwritten."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(checkpoint_path, 10)

        save_checkpoint(checkpoint_path, 20)

        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert data["last_page"] == 20


class TestFetchBooksIncremental:
    """Tests for fetch_books_incremental function."""

    def test_raises_error_when_no_languages_provided(
        self, tmp_path: Path, memory_parquet_store: InMemoryParquetStore
    ) -> None:
        """Test that empty languages list raises ValueError."""
        checkpoint = tmp_path / "checkpoint.json"

        with pytest.raises(ValueError, match="At least one language"):
            fetch_books_incremental(
                [],
                False,
                checkpoint,
                None,
                parquet_store=memory_parquet_store,
            )

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_fetches_single_page_successfully(
        self, mock_get: Mock, tmp_path: Path, memory_parquet_store: InMemoryParquetStore
    ) -> None:
        """Test successful single-page fetch."""
        checkpoint = tmp_path / "checkpoint.json"

        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Test Book",
                    "authors": [{"name": "Test Author"}],
                    "subjects": ["Fiction"],
                    "bookshelves": ["Adventure"],
                    "languages": ["en"],
                    "download_count": 100,
                    "media_type": "Text",
                    "copyright": True,
                }
            ],
        }
        mock_get.return_value = mock_response

        df = fetch_books_incremental(
            ["en"],
            True,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["title"] == "Test Book"
        assert df.iloc[0]["authors"] == "Test Author"
        mock_get.assert_called_once()

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_filters_english_only_when_flag_set(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that english_only flag filters multi-language books."""
        checkpoint = tmp_path / "checkpoint.json"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "English Only",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 50,
                    "media_type": "Text",
                    "copyright": False,
                },
                {
                    "id": 2,
                    "title": "Bilingual Book",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en", "fr"],
                    "download_count": 30,
                    "media_type": "Text",
                    "copyright": False,
                },
            ],
        }
        mock_get.return_value = mock_response

        df = fetch_books_incremental(
            ["en"],
            True,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1
        assert df.iloc[0]["id"] == 1

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_includes_multi_language_when_flag_not_set(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that multi-language books are included when english_only=False."""
        checkpoint = tmp_path / "checkpoint.json"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 2,
                    "title": "Bilingual",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en", "fr"],
                    "download_count": 20,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }
        mock_get.return_value = mock_response

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1
        assert df.iloc[0]["id"] == 2

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_follows_pagination_links(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that pagination links are followed correctly."""
        checkpoint = tmp_path / "checkpoint.json"

        # Mock two pages
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "count": 2,
            "next": "https://example.com/page2",
            "results": [
                {
                    "id": 1,
                    "title": "Book 1",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 10,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }

        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "count": 2,
            "next": None,
            "results": [
                {
                    "id": 2,
                    "title": "Book 2",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 20,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }

        mock_get.side_effect = [mock_response1, mock_response2]

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 2
        assert df.iloc[0]["id"] == 1
        assert df.iloc[1]["id"] == 2
        assert mock_get.call_count == 2

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    @patch("scripts.production.build_gutenberg_id_list.time.sleep")
    def test_retries_on_rate_limit(
        self,
        mock_sleep: Mock,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that 429 rate limit errors trigger retries."""
        checkpoint = tmp_path / "checkpoint.json"

        # First call: 429, second call: success
        mock_rate_limited = MagicMock()
        mock_rate_limited.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=Mock(status_code=429)
        )

        mock_success = MagicMock()
        mock_success.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Book",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 5,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }

        mock_get.side_effect = [mock_rate_limited, mock_success]

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1
        assert mock_sleep.call_count == 1  # Should have slept once

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    @patch("scripts.production.build_gutenberg_id_list.time.sleep")
    def test_gives_up_after_max_retries(
        self,
        mock_sleep: Mock,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that function returns partial data after max retries."""
        checkpoint = tmp_path / "checkpoint.json"

        # Always return 429
        mock_rate_limited = MagicMock()
        mock_rate_limited.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=Mock(status_code=429)
        )
        mock_get.return_value = mock_rate_limited

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 0  # No data fetched
        assert mock_get.call_count == 5  # MAX_RETRIES

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_resumes_from_checkpoint(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that fetch resumes from saved checkpoint."""
        checkpoint = tmp_path / "checkpoint.json"

        # Save checkpoint at page 1
        save_checkpoint(checkpoint, 1)

        # Create existing parquet with one book
        existing_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "title": "Existing Book",
                    "authors": "",
                    "subjects": "",
                    "bookshelves": "",
                    "languages": "en",
                    "download_count": 10,
                    "media_type": "Text",
                    "copyright": False,
                }
            ]
        )
        memory_parquet_store.save(existing_df)

        # Mock: first call skips to checkpoint, second call gets new data
        mock_skip = MagicMock()
        mock_skip.json.return_value = {
            "count": 2,
            "next": "https://example.com/page2",
            "results": [],
        }

        mock_new_page = MagicMock()
        mock_new_page.json.return_value = {
            "count": 2,
            "next": None,
            "results": [
                {
                    "id": 2,
                    "title": "New Book",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 20,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }

        mock_get.side_effect = [mock_skip, mock_new_page]

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        # Should have both existing and new book
        assert len(df) == 2
        # pandas-stubs marks Series.to_list as unknown; cast for type checker only.
        id_values = cast("list[int]", df["id"].astype(int).to_list())  # type: ignore[reportUnknownMemberType]
        ids = set(id_values)
        assert 1 in ids
        assert 2 in ids

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_saves_checkpoint_after_each_page(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that checkpoint is saved after each successful page."""
        checkpoint = tmp_path / "checkpoint.json"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Book",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["en"],
                    "download_count": 5,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }
        mock_get.return_value = mock_response

        fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert data["last_page"] == 1

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_handles_missing_optional_fields(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test handling of books with missing optional fields."""
        checkpoint = tmp_path / "checkpoint.json"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Minimal Book",
                    # Missing authors, subjects, bookshelves, etc.
                    "languages": ["en"],
                }
            ],
        }
        mock_get.return_value = mock_response

        df = fetch_books_incremental(
            ["en"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1
        assert df.iloc[0]["authors"] == ""
        assert df.iloc[0]["subjects"] == ""

    @patch("scripts.production.build_gutenberg_id_list.requests.get")
    def test_normalizes_language_codes(
        self,
        mock_get: Mock,
        tmp_path: Path,
        memory_parquet_store: InMemoryParquetStore,
    ) -> None:
        """Test that language codes are normalized (lowercased, trimmed)."""
        checkpoint = tmp_path / "checkpoint.json"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "id": 1,
                    "title": "Book",
                    "authors": [],
                    "subjects": [],
                    "bookshelves": [],
                    "languages": ["EN"],  # Uppercase in API response
                    "download_count": 5,
                    "media_type": "Text",
                    "copyright": False,
                }
            ],
        }
        mock_get.return_value = mock_response

        # Query with mixed case
        df = fetch_books_incremental(
            [" EN ", "fr"],
            False,
            checkpoint,
            None,
            parquet_store=memory_parquet_store,
        )

        assert len(df) == 1  # Should match despite case difference

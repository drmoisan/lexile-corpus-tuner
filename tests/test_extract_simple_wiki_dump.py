"""Unit tests for lexile_corpus_tuner/pipeline_scripts/extract_simple_wiki_dump.py.

Tests the Simple Wikipedia XML dump extractor following unit-test-policy.md.
"""

from __future__ import annotations

import io
import json
import sys
from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    extract_simple_wiki_dump as eswd,
)

NAMESPACE = eswd.NAMESPACE
iter_articles = eswd.iter_articles
open_dump = eswd.open_dump

if TYPE_CHECKING:
    from pathlib import Path


class TestOpenDump:
    """Tests for open_dump function."""

    @patch("bz2.open", new_callable=MagicMock)
    def test_opens_bz2_file(self, mock_bz2_open: Mock, tmp_path: Path) -> None:
        """Test that bz2 files are opened with bz2.open."""
        test_data = b"test content"
        bz2_file = tmp_path / "test.xml.bz2"

        # Mock bz2.open to return a BytesIO stream
        mock_stream = io.BytesIO(test_data)
        mock_bz2_open.return_value.__enter__.return_value = mock_stream

        with open_dump(bz2_file) as stream:
            content = stream.read()
            assert content == test_data

        # Verify bz2.open was called
        mock_bz2_open.assert_called_once()

    @patch("pathlib.Path.open", new_callable=MagicMock)
    def test_opens_regular_file(self, mock_open: Mock, tmp_path: Path) -> None:
        """Test that non-bz2 files are opened normally."""
        test_data = b"test content"
        regular_file = tmp_path / "test.xml"

        # Mock Path.open to return a BytesIO stream
        mock_stream = io.BytesIO(test_data)
        mock_open.return_value.__enter__.return_value = mock_stream

        with open_dump(regular_file) as stream:
            content = stream.read()
            assert content == test_data

        # Verify Path.open was called
        mock_open.assert_called_once()


class TestIterArticles:
    """Tests for iter_articles function."""

    def test_extracts_single_article(self) -> None:
        """Test extraction of a single valid article."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Test Article</title>
            <ns>0</ns>
            <id>123</id>
            <revision>
              <text>Test article content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["id"] == 123
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["text"] == "Test article content."
        assert articles[0]["source_id"] == "simple_wiki"

    def test_skips_non_main_namespace(self) -> None:
        """Test that articles in non-main namespace (ns != 0) are skipped."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Talk:Something</title>
            <ns>1</ns>
            <id>456</id>
            <revision>
              <text>Talk page content.</text>
            </revision>
          </page>
          <page>
            <title>Main Article</title>
            <ns>0</ns>
            <id>789</id>
            <revision>
              <text>Main content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["id"] == 789
        assert articles[0]["title"] == "Main Article"

    def test_skips_redirects(self) -> None:
        """Test that redirect pages are skipped."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Redirect Page</title>
            <ns>0</ns>
            <id>111</id>
            <redirect title="Target Page" />
            <revision>
              <text>#REDIRECT [[Target Page]]</text>
            </revision>
          </page>
          <page>
            <title>Normal Page</title>
            <ns>0</ns>
            <id>222</id>
            <revision>
              <text>Normal content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["id"] == 222
        assert articles[0]["title"] == "Normal Page"

    def test_handles_missing_text_element(self) -> None:
        """Test handling of pages without text element."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>No Text</title>
            <ns>0</ns>
            <id>333</id>
            <revision>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["text"] == ""

    def test_handles_missing_revision_element(self) -> None:
        """Test handling of pages without revision element."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>No Revision</title>
            <ns>0</ns>
            <id>444</id>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["text"] == ""

    def test_handles_invalid_page_id(self) -> None:
        """Test handling of non-numeric page IDs."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Bad ID</title>
            <ns>0</ns>
            <id>not_a_number</id>
            <revision>
              <text>Content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["id"] == 0

    def test_handles_missing_page_id(self) -> None:
        """Test handling of missing page ID element."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>No ID</title>
            <ns>0</ns>
            <revision>
              <text>Content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["id"] == 0

    def test_handles_missing_title(self) -> None:
        """Test handling of missing title element."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <ns>0</ns>
            <id>555</id>
            <revision>
              <text>Content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert articles[0]["title"] == ""

    def test_processes_multiple_articles(self) -> None:
        """Test extraction of multiple valid articles."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Article 1</title>
            <ns>0</ns>
            <id>1</id>
            <revision>
              <text>Content 1.</text>
            </revision>
          </page>
          <page>
            <title>Article 2</title>
            <ns>0</ns>
            <id>2</id>
            <revision>
              <text>Content 2.</text>
            </revision>
          </page>
          <page>
            <title>Article 3</title>
            <ns>0</ns>
            <id>3</id>
            <revision>
              <text>Content 3.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 3
        assert [a["id"] for a in articles] == [1, 2, 3]
        assert [a["title"] for a in articles] == ["Article 1", "Article 2", "Article 3"]

    def test_handles_unicode_content(self) -> None:
        """Test handling of Unicode characters in article content."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Français</title>
            <ns>0</ns>
            <id>666</id>
            <revision>
              <text>Contenu en français: café, naïve, 中文</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        articles = list(iter_articles(stream))

        assert len(articles) == 1
        assert "café" in articles[0]["text"]
        assert "中文" in articles[0]["text"]

    def test_returns_iterator_not_list(self) -> None:
        """Test that iter_articles returns an iterator for memory efficiency."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Test</title>
            <ns>0</ns>
            <id>1</id>
            <revision>
              <text>Content.</text>
            </revision>
          </page>
        </mediawiki>
        """
        stream = BytesIO(xml.encode("utf-8"))
        result = iter_articles(stream)

        # Should be an iterator, not a list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestMainIntegration:
    """Integration tests for the main function."""

    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_simple_wiki_dump.open_dump"
    )
    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_creates_output_file(
        self,
        mock_exists: Mock,
        mock_mkdir: Mock,
        mock_path_open: Mock,
        mock_open_dump: Mock,
        tmp_path: Path,
    ) -> None:
        """Test that main creates output JSONL file with articles."""
        # Create test XML dump (in memory)
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Test Article</title>
            <ns>0</ns>
            <id>999</id>
            <revision>
              <text>Test content for integration.</text>
            </revision>
          </page>
        </mediawiki>
        """
        dump_file = tmp_path / "test_dump.xml"
        output_file = tmp_path / "output.jsonl"

        # Mock dump stream
        mock_dump_stream = io.BytesIO(xml.encode("utf-8"))
        mock_open_dump.return_value.__enter__.return_value = mock_dump_stream

        # Capture output written
        written_content: list[str] = []

        def capture_write(content: str) -> None:
            written_content.append(content)

        mock_output_file = MagicMock()
        mock_output_file.write.side_effect = capture_write
        mock_path_open.return_value.__enter__.return_value = mock_output_file

        # Import and call main with mocked argv
        from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
            extract_simple_wiki_dump,
        )

        sys.argv = [
            "extract_simple_wiki_dump.py",
            "--dump",
            str(dump_file),
            "--output",
            str(output_file),
        ]

        extract_simple_wiki_dump.main()

        # Verify output content
        assert len(written_content) > 0
        # Find the JSON line (not the newline)
        json_lines: list[str] = [c for c in written_content if c and c != "\n"]
        assert len(json_lines) >= 1
        article = json.loads(json_lines[0])
        assert article["id"] == 999
        assert article["title"] == "Test Article"
        assert "integration" in article["text"]

    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_simple_wiki_dump.open_dump"
    )
    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_respects_max_articles_limit(
        self,
        mock_exists: Mock,
        mock_mkdir: Mock,
        mock_path_open: Mock,
        mock_open_dump: Mock,
        tmp_path: Path,
    ) -> None:
        """Test that --max-articles limits output."""
        xml = f"""<?xml version="1.0"?>
        <mediawiki xmlns="{NAMESPACE[1:-1]}">
          <page>
            <title>Article 1</title>
            <ns>0</ns>
            <id>1</id>
            <revision>
              <text>Content 1.</text>
            </revision>
          </page>
          <page>
            <title>Article 2</title>
            <ns>0</ns>
            <id>2</id>
            <revision>
              <text>Content 2.</text>
            </revision>
          </page>
          <page>
            <title>Article 3</title>
            <ns>0</ns>
            <id>3</id>
            <revision>
              <text>Content 3.</text>
            </revision>
          </page>
        </mediawiki>
        """
        dump_file = tmp_path / "test_dump.xml"
        output_file = tmp_path / "output.jsonl"

        # Mock dump stream
        mock_dump_stream = io.BytesIO(xml.encode("utf-8"))
        mock_open_dump.return_value.__enter__.return_value = mock_dump_stream

        # Capture output written
        written_content: list[str] = []

        def capture_write(content: str) -> None:
            written_content.append(content)

        mock_output_file = MagicMock()
        mock_output_file.write.side_effect = capture_write
        mock_path_open.return_value.__enter__.return_value = mock_output_file

        from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
            extract_simple_wiki_dump,
        )

        sys.argv = [
            "extract_simple_wiki_dump.py",
            "--dump",
            str(dump_file),
            "--output",
            str(output_file),
            "--max-articles",
            "2",
        ]

        extract_simple_wiki_dump.main()

        # Should write 2 articles total.
        # Each one produces JSON plus newline (two writes per article).
        json_lines: list[str] = [c for c in written_content if c and c != "\n"]
        assert len(json_lines) == 2

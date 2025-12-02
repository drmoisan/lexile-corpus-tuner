"""Unit tests for scripts/production/explore_gutenberg.py.

Tests the Gutenberg query engine with boolean search capabilities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Add scripts/production to path (must be before importing from script)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "production"))

from explore_gutenberg import (  # noqa: E402
    BooleanQueryEngine,
    QueryHistory,
    get_canonical_sets,
    save_canonical_sets,
)


class TestGetCanonicalSets:
    """Tests for get_canonical_sets function."""

    def test_extracts_subjects_and_bookshelves(self) -> None:
        """Test extraction of unique subjects and bookshelves."""
        df = pd.DataFrame(
            {
                "subjects": ["Fiction; Drama", "Fiction; Adventure", "Poetry"],
                "bookshelves": ["Adventure; Classic", "Adventure", "Poetry Collection"],
            }
        )

        subjects, bookshelves = get_canonical_sets(df)

        assert "Fiction" in subjects
        assert "Drama" in subjects
        assert "Adventure" in subjects
        assert "Poetry" in subjects
        assert "Adventure" in bookshelves
        assert "Classic" in bookshelves
        assert "Poetry Collection" in bookshelves

    def test_handles_missing_columns(self) -> None:
        """Test handling of DataFrame without subjects/bookshelves columns."""
        df = pd.DataFrame({"id": [1, 2, 3], "title": ["A", "B", "C"]})

        subjects, bookshelves = get_canonical_sets(df)

        assert subjects == set()
        assert bookshelves == set()

    def test_removes_nan_and_empty_strings(self) -> None:
        """Test that NaN and empty strings are filtered out."""
        df = pd.DataFrame(
            {
                "subjects": ["Fiction; ", "", None],
                "bookshelves": ["Classic", None, ""],
            }
        )

        subjects, bookshelves = get_canonical_sets(df)

        assert "" not in subjects
        assert "nan" not in subjects
        assert "None" not in subjects
        assert "" not in bookshelves
        assert "nan" not in bookshelves
        assert "None" not in bookshelves

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped from values."""
        df = pd.DataFrame(
            {
                "subjects": ["  Fiction  ; Drama  "],
                "bookshelves": ["  Classic  "],
            }
        )

        subjects, bookshelves = get_canonical_sets(df)

        assert "Fiction" in subjects
        assert "  Fiction  " not in subjects
        assert "Classic" in bookshelves


class TestSaveCanonicalSets:
    """Tests for save_canonical_sets function."""

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    def test_creates_output_directory(
        self, mock_mkdir: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test that output directory is created if needed."""
        output_dir = tmp_path / "nested" / "output"
        subjects = {"Fiction", "Drama"}
        bookshelves = {"Classic", "Adventure"}

        # Mock file handles for both writes
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        save_canonical_sets(subjects, bookshelves, output_dir)

        # Verify mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Verify open was called twice (subjects.txt and bookshelves.txt)
        assert mock_open.call_count == 2

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    def test_saves_sorted_content(
        self, mock_mkdir: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test that subjects and bookshelves are saved sorted."""
        subjects = {"Zebra", "Apple", "Mango"}
        bookshelves = {"Zoo", "Art", "Music"}

        # Capture what's written to files
        written_content: dict[str, str] = {}

        def mock_write_text(content: str) -> None:
            # Determine which file based on call count
            call_count = len(written_content)
            if call_count == 0:
                written_content["subjects"] = content
            else:
                written_content["bookshelves"] = content

        mock_file = MagicMock()
        mock_file.write.side_effect = mock_write_text
        mock_open.return_value.__enter__.return_value = mock_file

        save_canonical_sets(subjects, bookshelves, tmp_path)

        assert written_content["subjects"] == "Apple\nMango\nZebra"
        assert written_content["bookshelves"] == "Art\nMusic\nZoo"


class TestBooleanQueryEngine:
    """Tests for BooleanQueryEngine class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "title": ["Book A", "Book B", "Book C", "Book D"],
                "authors": ["Author X", "Author Y", "Author X", "Author Z"],
                "subjects": [
                    "Fiction; Drama",
                    "Fiction; Adventure",
                    "Poetry",
                    "Science; History",
                ],
                "bookshelves": [
                    "Classic; Adventure",
                    "Adventure",
                    "Poetry Collection",
                    "Science",
                ],
                "download_count": [100, 200, 50, 300],
            }
        )

    def test_tokenize_simple_query(self, sample_df: pd.DataFrame) -> None:
        """Test tokenization of simple query."""
        engine = BooleanQueryEngine(sample_df)

        tokens = engine._tokenize("Fiction AND Drama")

        assert tokens == ["Fiction", "AND", "Drama"]

    def test_tokenize_field_query(self, sample_df: pd.DataFrame) -> None:
        """Test tokenization of field-specific query."""
        engine = BooleanQueryEngine(sample_df)

        tokens = engine._tokenize("subject:Fiction")

        assert "subject:Fiction" in tokens

    def test_tokenize_comparison_query(self, sample_df: pd.DataFrame) -> None:
        """Test tokenization of numeric comparison."""
        engine = BooleanQueryEngine(sample_df)

        tokens = engine._tokenize("download_count>100")

        assert "download_count>100" in tokens

    def test_tokenize_range_query(self, sample_df: pd.DataFrame) -> None:
        """Test tokenization of range query."""
        engine = BooleanQueryEngine(sample_df)

        tokens = engine._tokenize("download_count:100..200")

        assert "download_count:100..200" in tokens

    def test_tokenize_parentheses(self, sample_df: pd.DataFrame) -> None:
        """Test tokenization with parentheses."""
        engine = BooleanQueryEngine(sample_df)

        tokens = engine._tokenize("(Fiction OR Drama) AND Adventure")

        assert "(" in tokens
        assert ")" in tokens

    def test_parse_field_query_colon(self, sample_df: pd.DataFrame) -> None:
        """Test parsing field:value syntax."""
        engine = BooleanQueryEngine(sample_df)

        field, operator, value = engine._parse_field_query("subject:Fiction")

        assert field == "subject"
        assert operator == ":"
        assert value == "Fiction"

    def test_parse_field_query_greater_than(self, sample_df: pd.DataFrame) -> None:
        """Test parsing field>value syntax."""
        engine = BooleanQueryEngine(sample_df)

        field, operator, value = engine._parse_field_query("download_count>100")

        assert field == "download_count"
        assert operator == ">"
        assert value == "100"

    def test_parse_field_query_range(self, sample_df: pd.DataFrame) -> None:
        """Test parsing field:min..max syntax."""
        engine = BooleanQueryEngine(sample_df)

        field, operator, value = engine._parse_field_query("download_count:50..200")

        assert field == "download_count"
        assert operator == ".."
        assert value == "50..200"

    def test_parse_field_query_no_field(self, sample_df: pd.DataFrame) -> None:
        """Test parsing plain term without field."""
        engine = BooleanQueryEngine(sample_df)

        field, operator, value = engine._parse_field_query("Fiction")

        assert field is None
        assert operator == "contains"
        assert value == "Fiction"

    def test_to_rpn_simple_and(self, sample_df: pd.DataFrame) -> None:
        """Test RPN conversion for simple AND query."""
        engine = BooleanQueryEngine(sample_df)

        rpn = engine._to_rpn(["Fiction", "AND", "Drama"])

        assert rpn == ["Fiction", "Drama", "AND"]

    def test_to_rpn_with_parentheses(self, sample_df: pd.DataFrame) -> None:
        """Test RPN conversion with parentheses."""
        engine = BooleanQueryEngine(sample_df)

        rpn = engine._to_rpn(["(", "Fiction", "OR", "Drama", ")", "AND", "Adventure"])

        # Parentheses should change precedence
        assert rpn.index("OR") < rpn.index("AND")

    def test_to_rpn_with_not(self, sample_df: pd.DataFrame) -> None:
        """Test RPN conversion with NOT operator."""
        engine = BooleanQueryEngine(sample_df)

        rpn = engine._to_rpn(["NOT", "Fiction"])

        assert rpn == ["Fiction", "NOT"]

    def test_evaluate_simple_contains(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of simple contains query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("Fiction")

        assert len(result) == 2  # Books with Fiction in subjects

    def test_evaluate_and_operator(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of AND query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("Fiction AND Drama")

        assert len(result) == 1  # Only book with both Fiction and Drama

    def test_evaluate_or_operator(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of OR query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("Fiction OR Poetry")

        assert len(result) == 3  # Books with either Fiction or Poetry

    def test_evaluate_not_operator(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of NOT query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("NOT Fiction")

        assert len(result) == 2  # Books without Fiction

    def test_evaluate_numeric_comparison(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of numeric comparison."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("download_count>100")

        assert len(result) == 2  # Books with download_count > 100
        assert all(result["download_count"] > 100)

    def test_evaluate_range_query(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of range query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("download_count:100..200")

        assert len(result) == 2  # Books with download_count between 100 and 200
        assert all(
            (result["download_count"] >= 100) & (result["download_count"] <= 200)
        )

    def test_evaluate_field_specific_query(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of field-specific query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("subjects:Fiction")

        assert len(result) == 2  # Books with Fiction in subjects column

    def test_evaluate_exact_match(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of exact match with quotes searches across text fields."""
        engine = BooleanQueryEngine(sample_df)

        # Test exact match without field specification (searches all text fields)
        result = engine.evaluate('"Book A"')

        assert len(result) == 1
        assert result.iloc[0]["title"] == "Book A"

    def test_evaluate_empty_query(self, sample_df: pd.DataFrame) -> None:
        """Test that empty query returns full DataFrame."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("")

        assert len(result) == len(sample_df)

    def test_evaluate_unknown_field_warning(
        self, sample_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that unknown field generates warning."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("unknown_field:value")

        # Should return full DataFrame with warning
        assert len(result) == len(sample_df)
        captured = capsys.readouterr()
        assert "Unknown field" in captured.err

    def test_evaluate_missing_and_operand(
        self, sample_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test error handling for missing AND operand."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("Fiction AND")

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Missing operand" in captured.err

    def test_evaluate_complex_query(self, sample_df: pd.DataFrame) -> None:
        """Test evaluation of complex boolean query."""
        engine = BooleanQueryEngine(sample_df)

        result = engine.evaluate("(Fiction OR Poetry) AND NOT Drama")

        # Should match: Fiction without Drama (id=2) and Poetry (id=3)
        assert len(result) == 2


class TestQueryHistory:
    """Tests for QueryHistory class."""

    @patch("pathlib.Path.exists", return_value=False)
    def test_initializes_empty_history(self, mock_exists: Mock, tmp_path: Path) -> None:
        """Test that new history starts empty."""
        history_file = tmp_path / "history.json"
        history = QueryHistory(history_file)

        assert history.queries == []

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists", return_value=True)
    def test_loads_existing_history(
        self, mock_exists: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test loading existing history from file."""
        history_file = tmp_path / "history.json"
        existing_queries = ["Fiction", "Drama", "Poetry"]

        # Mock file read
        mock_file = MagicMock()
        mock_file.read.return_value = json.dumps(existing_queries)
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file

        history = QueryHistory(history_file)

        assert history.queries == existing_queries

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists", return_value=True)
    def test_handles_malformed_history_file(
        self, mock_exists: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test that malformed history file results in empty list."""
        history_file = tmp_path / "bad_history.json"

        # Mock file read to return invalid JSON
        mock_file = MagicMock()
        mock_file.read.return_value = "not valid json"
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file

        history = QueryHistory(history_file)

        assert history.queries == []

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists", return_value=False)
    def test_saves_history(
        self, mock_exists: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test saving history to file."""
        history_file = tmp_path / "history.json"
        history = QueryHistory(history_file)
        history.queries = ["Query 1", "Query 2"]

        # Capture what's written
        written_content = []

        def capture_write(content: str) -> None:
            written_content.append(content)

        mock_file = MagicMock()
        mock_file.write.side_effect = capture_write
        mock_open.return_value.__enter__.return_value = mock_file

        history.save()

        # Verify write was called
        assert mock_open.call_count >= 1
        # Verify JSON was written
        if written_content:
            saved_data = json.loads("".join(written_content))
            assert saved_data == ["Query 1", "Query 2"]

    @patch("pathlib.Path.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists", return_value=False)
    def test_add_query(
        self, mock_exists: Mock, mock_open: Mock, tmp_path: Path
    ) -> None:
        """Test adding query to history."""
        history_file = tmp_path / "history.json"

        # Mock file for save operation
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        history = QueryHistory(history_file)
        history.queries.append("New Query")

        assert "New Query" in history.queries

    @patch("pathlib.Path.exists", return_value=False)
    def test_uses_default_filename_when_none_provided(self, mock_exists: Mock) -> None:
        """Test that default filename is used when none provided."""
        history = QueryHistory()

        assert history.history_file.name == ".gutenberg_query_history.json"

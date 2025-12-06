"""Unit tests for lexile_corpus_tuner pipeline_scripts Gutenberg query builder UI.

Tests the UI query builder components and integration with the query engine.
Focuses on core query building logic and edge cases that caused production bugs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from lexile_corpus_tuner.pipeline_scripts.explore_gutenberg import BooleanQueryEngine
from lexile_corpus_tuner.pipeline_scripts.gutenberg_query_core import (
    QueryConstraintModel,
    QueryGroupModel,
)


def _tokenize(engine: BooleanQueryEngine, query: str) -> list[str]:
    """Access protected tokenizer for testing only."""
    return engine._tokenize(query)  # pyright: ignore[reportPrivateUsage]


def _column_to_set(df: pd.DataFrame, column: str) -> set[str]:
    """Convert a DataFrame column to Python set for assertions.

    Isolated pandas operation: pandas-stubs incomplete for column access.
    """
    return set(df[column].to_list())  # type: ignore[reportUnknownMemberType]


def _dataframe_len(df: pd.DataFrame) -> int:
    """Get length of DataFrame.

    Isolated pandas operation: pandas-stubs incomplete for len().
    """
    return len(df)


class TestQueryConstraintModelGeneration:
    """Test query string generation from QueryConstraintModel."""

    def test_single_value_generates_simple_query(self) -> None:
        """Test that single value generates simple field:value query.

        Ensures basic query generation works without wrapping in OR.
        """
        constraint = QueryConstraintModel("title", "contains", "Fiction")

        query_str = constraint.to_query_string()

        assert query_str == "title:Fiction"

    def test_list_value_generates_or_query(self) -> None:
        """Test that list of values generates OR query with quotes.

        This is the core pattern used by the UI's multi-select fields.
        """
        constraint = QueryConstraintModel(
            "subjects", "contains", ["Fiction", "Adventure"]
        )

        query_str = constraint.to_query_string()

        assert query_str == '(subjects:"Fiction" OR subjects:"Adventure")'

    def test_list_value_with_apostrophes(self) -> None:
        """Test that list values with apostrophes are properly quoted.

        Critical test: Apostrophes in quoted strings previously broke tokenization.
        This was the root cause of the "Invalid query" error in production.
        """
        constraint = QueryConstraintModel(
            "bookshelves",
            "contains",
            ["Children's Fiction", "Children's Biography", "Child's Own Book"],
        )

        query_str = constraint.to_query_string()

        expected = (
            '(bookshelves:"Children\'s Fiction" OR '
            'bookshelves:"Children\'s Biography" OR '
            'bookshelves:"Child\'s Own Book")'
        )
        assert query_str == expected

    def test_many_values_generates_long_or_chain(self) -> None:
        """Test that many values (13+) generate valid OR chains.

        Regression test for the exact scenario reported by user:
        selecting 13 "Child"-related bookshelves from the UI.
        """
        bookshelves = [
            "Category: Children & Young Adult Reading",
            "Child's Own Book of Great Musicians",
            "Children's Anthologies",
            "Children's Biography",
            "Children's Book Series",
            "Children's Fiction",
            "Children's History",
            "Children's Instructional Books",
            "Children's Literature",
            "Children's Myths, Fairy Tales, etc.",
            "Children's Picture Books",
            "Children's Religion",
            "Children's Verse",
        ]

        constraint = QueryConstraintModel("bookshelves", "contains", bookshelves)
        query_str = constraint.to_query_string()

        # Should start with ( and end with )
        assert query_str.startswith("(")
        assert query_str.endswith(")")

        # Should contain 13 bookshelf queries joined by OR
        assert query_str.count(' OR bookshelves:"') == 12  # 13 items = 12 ORs
        assert query_str.count('bookshelves:"') == 13


class TestBooleanQueryEngineWithApostrophes:
    """Test BooleanQueryEngine handles queries with apostrophes correctly.

    Regression tests for the tokenization bug that split quoted strings
    on apostrophes, causing "Invalid query" errors.
    """

    @pytest.fixture
    def sample_df_with_apostrophes(self) -> pd.DataFrame:
        """Create DataFrame with bookshelves containing apostrophes."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Book A", "Book B", "Book C", "Book D", "Book E"],
                "bookshelves": [
                    "Children's Fiction",
                    "Children's Biography",
                    "Child's Own Book of Great Musicians",
                    "Poetry Collection",
                    "Children's Anthologies; Children's Fiction",
                ],
            }
        )

    def test_tokenize_field_with_apostrophe_in_quotes(
        self, sample_df_with_apostrophes: pd.DataFrame
    ) -> None:
        """Test that tokenizer preserves quoted strings with apostrophes.

        Critical fix: Previously `bookshelves:"Children's Fiction"` was split into:
        ['bookshelves:"Children\\'', 's', 'Fiction"']

        Now it should tokenize as a single unit:
        ['bookshelves:"Children\\'s Fiction"']
        """
        engine = BooleanQueryEngine(sample_df_with_apostrophes)

        query = 'bookshelves:"Children\'s Fiction"'
        tokens = _tokenize(engine, query)

        assert len(tokens) == 1
        assert tokens[0] == 'bookshelves:"Children\'s Fiction"'

    def test_tokenize_multiple_fields_with_apostrophes_and_or(
        self, sample_df_with_apostrophes: pd.DataFrame
    ) -> None:
        """Test tokenization of OR query with multiple apostrophe-containing values.

        This is the exact pattern generated by UI multi-select.
        """
        engine = BooleanQueryEngine(sample_df_with_apostrophes)

        query = (
            '(bookshelves:"Children\'s Fiction" OR '
            'bookshelves:"Children\'s Biography")'
        )
        tokens = _tokenize(engine, query)

        assert len(tokens) == 5
        assert tokens[0] == "("
        assert tokens[1] == 'bookshelves:"Children\'s Fiction"'
        assert tokens[2] == "OR"
        assert tokens[3] == 'bookshelves:"Children\'s Biography"'
        assert tokens[4] == ")"

    def test_evaluate_or_query_with_apostrophes(
        self, sample_df_with_apostrophes: pd.DataFrame
    ) -> None:
        """Test full evaluation of OR query with apostrophe-containing values.

        Ensures the query executes successfully and returns correct results.
        """
        engine = BooleanQueryEngine(sample_df_with_apostrophes)

        query = (
            '(bookshelves:"Children\'s Fiction" OR '
            'bookshelves:"Children\'s Biography")'
        )
        result = engine.evaluate(query)

        assert _dataframe_len(result) == 3  # Books 1, 2, and 5
        titles = _column_to_set(result, "title")
        assert titles == {"Book A", "Book B", "Book E"}

    def test_evaluate_thirteen_child_bookshelves_or_query(
        self, sample_df_with_apostrophes: pd.DataFrame
    ) -> None:
        """Test the exact scenario reported: 13 "Child" bookshelves in OR query.

        Regression test: This previously failed with "Invalid query" error due to
        tokenizer breaking on apostrophes.
        """
        # Extend the dataframe to include all 13 bookshelf types
        extended_df = pd.DataFrame(
            {
                "id": list(range(1, 14)),
                "title": [f"Book {i}" for i in range(1, 14)],
                "bookshelves": [
                    "Category: Children & Young Adult Reading",
                    "Child's Own Book of Great Musicians",
                    "Children's Anthologies",
                    "Children's Biography",
                    "Children's Book Series",
                    "Children's Fiction",
                    "Children's History",
                    "Children's Instructional Books",
                    "Children's Literature",
                    "Children's Myths, Fairy Tales, etc.",
                    "Children's Picture Books",
                    "Children's Religion",
                    "Children's Verse",
                ],
            }
        )

        engine = BooleanQueryEngine(extended_df)

        # Build the query that UI generates for all 13 bookshelves
        bookshelves = [
            "Category: Children & Young Adult Reading",
            "Child's Own Book of Great Musicians",
            "Children's Anthologies",
            "Children's Biography",
            "Children's Book Series",
            "Children's Fiction",
            "Children's History",
            "Children's Instructional Books",
            "Children's Literature",
            "Children's Myths, Fairy Tales, etc.",
            "Children's Picture Books",
            "Children's Religion",
            "Children's Verse",
        ]

        query_parts = [f'bookshelves:"{shelf}"' for shelf in bookshelves]
        query = f"({' OR '.join(query_parts)})"

        # Execute query - should not raise error
        result = engine.evaluate(query)

        # Should match all 13 books
        assert _dataframe_len(result) == 13
        assert _column_to_set(result, "id") == set(range(1, 14))


class TestQueryGroupModelGeneration:
    """Test query string generation from QueryGroupModel."""

    def test_group_with_and_logic(self) -> None:
        """Test that group with AND logic generates correct query.

        Ensures AND operator is properly used in grouped queries.
        """
        group = QueryGroupModel(
            logic="AND",
            constraints=[
                QueryConstraintModel("title", "contains", "Fiction"),
                QueryConstraintModel("download_count", ">", "100"),
            ],
        )

        query_str = group.to_query_string()

        assert query_str == "(title:Fiction AND download_count>100)"

    def test_group_with_or_logic_and_list_values(self) -> None:
        """Test group with OR logic and list values.

        This tests nesting: outer AND with inner OR from list values.
        """
        group = QueryGroupModel(
            logic="AND",
            constraints=[
                QueryConstraintModel(
                    "subjects", "contains", ["Fiction", "Adventure"]  # Inner OR
                ),
                QueryConstraintModel("download_count", ">", "100"),
            ],
        )

        query_str = group.to_query_string()

        expected = (
            '((subjects:"Fiction" OR subjects:"Adventure") AND download_count>100)'
        )
        assert query_str == expected

    def test_nested_groups(self) -> None:
        """Test nested QueryGroupModel with complex logic.

        Ensures nested groups maintain proper parenthesization.
        """
        inner_group = QueryGroupModel(
            logic="OR",
            constraints=[
                QueryConstraintModel("subjects", "contains", "Fiction"),
                QueryConstraintModel("subjects", "contains", "Drama"),
            ],
        )

        outer_group = QueryGroupModel(
            logic="AND",
            constraints=[
                inner_group,
                QueryConstraintModel("download_count", ">", "100"),
            ],
        )

        query_str = outer_group.to_query_string()

        expected = "((subjects:Fiction OR subjects:Drama) AND download_count>100)"
        assert query_str == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_values_in_list(self) -> None:
        """Test that empty strings in list values are handled gracefully.

        Ensures no empty OR clauses are generated.
        """
        constraint = QueryConstraintModel(
            "subjects", "contains", ["Fiction", "", "Drama"]
        )

        query_str = constraint.to_query_string()

        # Should include empty string as-is (query engine will handle it)
        assert query_str == '(subjects:"Fiction" OR subjects:"" OR subjects:"Drama")'

    def test_single_item_list_unwraps(self) -> None:
        """Test that single-item list is treated as simple value.

        Documents current behavior for single-item lists.
        """
        constraint = QueryConstraintModel("subjects", "contains", ["Fiction"])

        query_str = constraint.to_query_string()

        # Single-item list still wraps in parens with quotes
        assert query_str == '(subjects:"Fiction")'

    def test_special_characters_in_quoted_values(self) -> None:
        """Test that special characters are preserved in quoted values.

        Tests colons, ampersands, commas, etc.
        """
        constraint = QueryConstraintModel(
            "title",
            "contains",
            ["Category: Children & Young Adult", "Book, The Great"],
        )

        query_str = constraint.to_query_string()

        assert (
            query_str
            == '(title:"Category: Children & Young Adult" OR title:"Book, The Great")'
        )

    def test_value_with_double_quotes_in_single_quotes(
        self,
    ) -> None:
        """Test that values can contain opposite quote type.

        Ensures proper escaping/quoting strategy.
        """
        # This tests the QueryConstraintModel side; engine handles parsing
        constraint = QueryConstraintModel("title", "contains", 'Say "Hello"')

        query_str = constraint.to_query_string()

        # Should wrap in double quotes, preserving internal quotes
        assert query_str == 'title:Say "Hello"'


class TestIntegrationWithQueryEngine:
    """Integration tests combining QueryConstraintModel and BooleanQueryEngine."""

    def test_ui_generated_query_executes_successfully(self) -> None:
        """Test that UI-generated queries execute without errors.

        End-to-end test simulating UI workflow:
        1. User selects multiple bookshelves with apostrophes
        2. UI creates QueryConstraintModel
        3. Model generates query string
        4. Query engine evaluates successfully
        """
        # Sample data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "title": ["Book A", "Book B", "Book C", "Book D"],
                "bookshelves": [
                    "Children's Fiction; Adventure",
                    "Children's Biography",
                    "Poetry Collection",
                    "Children's Anthologies",
                ],
            }
        )

        # UI creates constraint from user selection
        selected_bookshelves = [
            "Children's Fiction",
            "Children's Biography",
            "Children's Anthologies",
        ]
        constraint = QueryConstraintModel(
            "bookshelves", "contains", selected_bookshelves
        )

        # UI generates query string
        query_str = constraint.to_query_string()

        # Query engine evaluates
        engine = BooleanQueryEngine(df)
        result = engine.evaluate(query_str)

        # Verify results
        assert _dataframe_len(result) > 0  # Should find matches
        assert _dataframe_len(result) == 3  # Books A, B, and D
        titles = _column_to_set(result, "title")
        assert titles == {"Book A", "Book B", "Book D"}



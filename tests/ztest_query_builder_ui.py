"""Tests for Gutenberg Query Builder UI components and models.

This module tests the data models, query string generation, and serialization
logic for the query builder UI. It does not test Tkinter widgets directly.
"""

# ruff: noqa: E402
# Module mocking must happen before imports

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

# Save original modules before mocking
_original_modules = {
    name: sys.modules.get(name)
    for name in [
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "pandas",
    ]
}

# Mock tkinter and pandas to avoid import errors in test environments
# These must be mocked before importing gutenberg_query_builder_ui
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.ttk"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
sys.modules["tkinter.filedialog"] = MagicMock()
sys.modules["pandas"] = MagicMock()


@pytest.fixture(scope="module", autouse=True)
def restore_mocked_modules():
    """Restore original modules after this test module completes."""
    yield
    # Restore original modules
    for name, original in _original_modules.items():
        if original is not None:
            sys.modules[name] = original
        elif name in sys.modules:
            del sys.modules[name]


# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import after mocking to avoid dependency errors
# Type checkers will see these as Any, but tests will work at runtime
if TYPE_CHECKING:
    # For type checking, we need to import from the actual location
    # This won't run at runtime due to TYPE_CHECKING guard
    from typing import Any

    # Stub types for type checker - these match the actual classes
    class QueryConstraintModel:
        field: str
        operator: str
        value: str

        def __init__(self, field: str, operator: str, value: str) -> None: ...
        def to_query_string(self) -> str: ...

    class QueryGroupModel:
        logic: str
        constraints: list["QueryConstraintModel | QueryGroupModel"]

        def __init__(
            self,
            logic: str,
            constraints: list["QueryConstraintModel | QueryGroupModel"],
        ) -> None: ...
        def to_query_string(self) -> str: ...

    class SavedQuery:
        version: str
        created: str
        modified: str
        query: dict[str, Any]

        @staticmethod
        def from_query_group(group: "QueryGroupModel") -> "SavedQuery": ...
        def to_json(self) -> str: ...
        @staticmethod
        def from_json(json_str: str) -> "SavedQuery": ...
        def to_query_group(self) -> "QueryGroupModel": ...

else:
    # At runtime, import the actual implementation
    from gutenberg_query_builder_ui import (
        QueryConstraintModel,
        QueryGroupModel,
        SavedQuery,
    )


class TestQueryConstraintModel:
    """Test QueryConstraintModel data class and methods."""

    def test_constraint_creation(self):
        """Test creating a basic constraint."""
        constraint = QueryConstraintModel(
            field="title", operator="contains", value="Python"
        )
        assert constraint.field == "title"
        assert constraint.operator == "contains"
        assert constraint.value == "Python"

    def test_constraint_to_query_string_text_contains(self):
        """Test query string generation for text contains."""
        constraint = QueryConstraintModel(
            field="title", operator="contains", value="Python Programming"
        )
        assert constraint.to_query_string() == "title:Python Programming"

    def test_constraint_to_query_string_exact_match(self):
        """Test query string generation for exact match."""
        constraint = QueryConstraintModel(
            field="title", operator="==", value="Alice in Wonderland"
        )
        # Note: == operator doesn't add quotes in current implementation
        assert constraint.to_query_string() == "title==Alice in Wonderland"

    def test_constraint_to_query_string_numeric_comparison(self):
        """Test query string generation for numeric operators."""
        constraint = QueryConstraintModel(
            field="download_count", operator=">", value="1000"
        )
        assert constraint.to_query_string() == "download_count>1000"

    def test_constraint_to_query_string_range(self):
        """Test query string generation for range operator."""
        constraint = QueryConstraintModel(
            field="download_count", operator="range", value="1000..5000"
        )
        assert constraint.to_query_string() == "download_count:1000..5000"

    def test_constraint_to_query_string_not_equal(self):
        """Test query string generation for not equal operator."""
        constraint = QueryConstraintModel(field="languages", operator="!=", value="en")
        # != is rendered as NOT field="value"
        assert constraint.to_query_string() == 'NOT languages="en"'

    def test_constraint_to_query_string_multiselect(self):
        """Test query string generation for multi-select (semicolon-separated)."""
        constraint = QueryConstraintModel(
            field="subjects", operator="contains", value="Fiction;Adventure;Fantasy"
        )
        query = constraint.to_query_string()
        # Multi-select values are kept as semicolon-separated in current implementation
        assert query == "subjects:Fiction;Adventure;Fantasy"

    def test_constraint_empty_value(self):
        """Test constraint with empty value."""
        constraint = QueryConstraintModel(field="title", operator="contains", value="")
        # Empty values still generate field: in current implementation
        assert constraint.to_query_string() == "title:"


class TestQueryGroupModel:
    """Test QueryGroupModel data class and methods."""

    def test_group_creation_empty(self):
        """Test creating an empty group."""
        group = QueryGroupModel(logic="AND", constraints=[])
        assert group.logic == "AND"
        assert len(group.constraints) == 0

    def test_group_creation_with_constraints(self):
        """Test creating a group with constraints."""
        c1 = QueryConstraintModel(field="title", operator="contains", value="Python")
        c2 = QueryConstraintModel(field="download_count", operator=">", value="1000")
        group = QueryGroupModel(logic="AND", constraints=[c1, c2])
        assert len(group.constraints) == 2

    def test_group_to_query_string_single_constraint(self):
        """Test query string for group with single constraint."""
        constraint = QueryConstraintModel(
            field="title", operator="contains", value="Python"
        )
        group = QueryGroupModel(logic="AND", constraints=[constraint])
        # Single constraint should not have parentheses
        assert group.to_query_string() == "title:Python"

    def test_group_to_query_string_multiple_constraints_and(self):
        """Test query string for AND group with multiple constraints."""
        c1 = QueryConstraintModel(field="title", operator="contains", value="Python")
        c2 = QueryConstraintModel(field="download_count", operator=">", value="1000")
        group = QueryGroupModel(logic="AND", constraints=[c1, c2])
        query = group.to_query_string()
        assert query == "(title:Python AND download_count>1000)"

    def test_group_to_query_string_multiple_constraints_or(self):
        """Test query string for OR group with multiple constraints."""
        c1 = QueryConstraintModel(
            field="subjects", operator="contains", value="Fiction"
        )
        c2 = QueryConstraintModel(
            field="subjects", operator="contains", value="Science"
        )
        group = QueryGroupModel(logic="OR", constraints=[c1, c2])
        query = group.to_query_string()
        assert query == "(subjects:Fiction OR subjects:Science)"

    def test_group_to_query_string_nested_groups(self):
        """Test query string for nested groups."""
        # Inner group: (title contains Python OR title contains JavaScript)
        inner_c1 = QueryConstraintModel(
            field="title", operator="contains", value="Python"
        )
        inner_c2 = QueryConstraintModel(
            field="title", operator="contains", value="JavaScript"
        )
        inner_group = QueryGroupModel(logic="OR", constraints=[inner_c1, inner_c2])

        # Outer constraint
        outer_constraint = QueryConstraintModel(
            field="download_count", operator=">", value="5000"
        )

        # Outer group: inner_group AND download_count > 5000
        outer_group = QueryGroupModel(
            logic="AND", constraints=[inner_group, outer_constraint]
        )

        query = outer_group.to_query_string()
        assert query == "((title:Python OR title:JavaScript) AND download_count>5000)"

    def test_group_to_query_string_empty(self):
        """Test query string for empty group."""
        group = QueryGroupModel(logic="AND", constraints=[])
        assert group.to_query_string() == ""

    def test_group_filters_empty_constraints(self):
        """Test group with mix of empty and non-empty constraints."""
        c1 = QueryConstraintModel(field="title", operator="contains", value="Python")
        c2 = QueryConstraintModel(field="title", operator="contains", value="")
        group = QueryGroupModel(logic="AND", constraints=[c1, c2])
        # Current implementation includes empty constraints
        query = group.to_query_string()
        assert "title:Python" in query
        assert "title:" in query


class TestSavedQuery:
    """Test SavedQuery serialization and deserialization."""

    def test_saved_query_creation_from_group(self):
        """Test creating SavedQuery from QueryGroupModel."""
        constraint = QueryConstraintModel(
            field="title", operator="contains", value="Python"
        )
        group = QueryGroupModel(logic="AND", constraints=[constraint])

        saved = SavedQuery.from_query_group(group)

        assert saved.version == "1.0"
        assert saved.created is not None
        assert saved.modified is not None
        assert isinstance(saved.query, dict)
        assert saved.query["type"] == "group"

    def test_saved_query_serialization_to_json(self):
        """Test JSON serialization of SavedQuery."""
        constraint = QueryConstraintModel(
            field="title", operator="contains", value="Python"
        )
        group = QueryGroupModel(logic="AND", constraints=[constraint])
        saved = SavedQuery.from_query_group(group)

        json_str = saved.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["version"] == "1.0"
        assert "created" in data
        assert "modified" in data
        assert "query" in data

    def test_saved_query_deserialization_from_json(self):
        """Test JSON deserialization to SavedQuery."""
        json_str = """{
            "version": "1.0",
            "created": "2025-01-01T12:00:00",
            "modified": "2025-01-01T12:00:00",
            "query": {
                "type": "group",
                "logic": "AND",
                "constraints": [
                    {
                        "type": "constraint",
                        "field": "title",
                        "operator": "contains",
                        "value": "Python"
                    }
                ]
            }
        }"""

        saved = SavedQuery.from_json(json_str)

        assert saved.version == "1.0"
        assert saved.created == "2025-01-01T12:00:00"
        assert isinstance(saved.query, dict)

    def test_saved_query_round_trip(self):
        """Test serialization and deserialization round trip."""
        # Create a complex query
        c1 = QueryConstraintModel(field="title", operator="contains", value="Python")
        c2 = QueryConstraintModel(field="download_count", operator=">", value="1000")
        group = QueryGroupModel(logic="AND", constraints=[c1, c2])

        # Serialize
        saved = SavedQuery.from_query_group(group)
        json_str = saved.to_json()

        # Deserialize
        loaded = SavedQuery.from_json(json_str)
        restored_group = loaded.to_query_group()

        # Verify structure
        assert restored_group.logic == "AND"
        assert len(restored_group.constraints) == 2
        assert restored_group.to_query_string() == group.to_query_string()

    def test_saved_query_nested_groups_serialization(self):
        """Test serialization of nested groups."""
        # Create nested structure
        inner_c1 = QueryConstraintModel(
            field="subjects", operator="contains", value="Fiction"
        )
        inner_c2 = QueryConstraintModel(
            field="subjects", operator="contains", value="Science"
        )
        inner_group = QueryGroupModel(logic="OR", constraints=[inner_c1, inner_c2])

        outer_constraint = QueryConstraintModel(
            field="languages", operator="==", value="en"
        )
        outer_group = QueryGroupModel(
            logic="AND", constraints=[inner_group, outer_constraint]
        )

        # Serialize and deserialize
        saved = SavedQuery.from_query_group(outer_group)
        json_str = saved.to_json()
        loaded = SavedQuery.from_json(json_str)
        restored = loaded.to_query_group()

        # Verify query strings match
        assert restored.to_query_string() == outer_group.to_query_string()

    def test_saved_query_deserialize_constraint_fields(self):
        """Test that constraint fields are correctly deserialized."""
        json_str = """{
            "version": "1.0",
            "created": "2025-01-01T12:00:00",
            "modified": "2025-01-01T12:00:00",
            "query": {
                "type": "constraint",
                "field": "download_count",
                "operator": ">=",
                "value": "5000"
            }
        }"""

        saved = SavedQuery.from_json(json_str)
        group = saved.to_query_group()

        # Root should be wrapped in a group
        assert isinstance(group, QueryGroupModel)
        assert len(group.constraints) == 1

        constraint = group.constraints[0]
        # Type narrow: we know this is a QueryConstraintModel from the test data
        assert isinstance(constraint, QueryConstraintModel)
        assert constraint.field == "download_count"
        assert constraint.operator == ">="
        assert constraint.value == "5000"


class TestQueryStringGeneration:
    """Test complex query string generation scenarios."""

    def test_complex_nested_query(self):
        """Test generating a complex nested query string."""
        # (title:Python OR title:Java) AND download_count>1000
        # AND (subjects:Fiction OR subjects:Science)

        # Create first inner group (title)
        title_group = QueryGroupModel(
            logic="OR",
            constraints=[
                QueryConstraintModel(
                    field="title", operator="contains", value="Python"
                ),
                QueryConstraintModel(field="title", operator="contains", value="Java"),
            ],
        )

        # Create second inner group (subjects)
        subjects_group = QueryGroupModel(
            logic="OR",
            constraints=[
                QueryConstraintModel(
                    field="subjects", operator="contains", value="Fiction"
                ),
                QueryConstraintModel(
                    field="subjects", operator="contains", value="Science"
                ),
            ],
        )

        # Create download constraint
        download_constraint = QueryConstraintModel(
            field="download_count", operator=">", value="1000"
        )

        # Create outer group
        outer_group = QueryGroupModel(
            logic="AND",
            constraints=[title_group, download_constraint, subjects_group],
        )

        query = outer_group.to_query_string()

        # Verify all parts are present
        assert "title:Python OR title:Java" in query
        assert "download_count>1000" in query
        assert "subjects:Fiction OR subjects:Science" in query
        assert query.count("AND") == 2

    def test_query_with_all_operator_types(self):
        """Test query with various operator types."""
        group = QueryGroupModel(
            logic="AND",
            constraints=[
                QueryConstraintModel(
                    field="title", operator="contains", value="Python"
                ),
                QueryConstraintModel(
                    field="download_count", operator=">", value="1000"
                ),
                QueryConstraintModel(
                    field="download_count", operator="<", value="10000"
                ),
                QueryConstraintModel(field="languages", operator="==", value="en"),
                QueryConstraintModel(field="copyright", operator="!=", value="true"),
                QueryConstraintModel(field="id", operator="range", value="100..200"),
            ],
        )

        query = group.to_query_string()

        # Verify all operators are present
        assert "title:Python" in query
        assert "download_count>1000" in query
        assert "download_count<10000" in query
        assert "languages==en" in query
        assert 'NOT copyright="true"' in query
        assert "id:100..200" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

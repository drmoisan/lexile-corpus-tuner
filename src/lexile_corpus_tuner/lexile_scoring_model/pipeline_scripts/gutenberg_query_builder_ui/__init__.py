"""Gutenberg Query Builder - Visual Query Construction UI.

This module provides a lightweight Tkinter-based GUI for building and executing
queries against the Gutenberg metadata database. It supports:
- Visual query construction with drag-and-drop
- Field-specific operators and values
- Grouping with AND/OR logic
- Multi-select for subjects/bookshelves
- Query persistence (save/load)
- Results export (CSV/Parquet)

Example usage:
    import tkinter as tk
    from gutenberg_query_builder_ui import main

    main()

Or programmatically:
    import tkinter as tk
    from gutenberg_query_builder_ui import QueryBuilderApp

    root = tk.Tk()
    app = QueryBuilderApp(root)
    app.run()
"""

from __future__ import annotations

import tkinter as tk

from .app import QueryBuilderApp
from .constants import (
    BOOLEAN_OPERATORS,
    FIELD_TYPES,
    NUMERIC_OPERATORS,
    PARQUET_PATH,
    TEXT_OPERATORS,
    WINDOW_SIZE,
    WINDOW_TITLE,
)
from .widgets import QueryConstraintWidget, QueryGroupWidget, ToolTip

__all__ = [
    "main",
    "QueryBuilderApp",
    "QueryConstraintWidget",
    "QueryGroupWidget",
    "ToolTip",
    "FIELD_TYPES",
    "TEXT_OPERATORS",
    "NUMERIC_OPERATORS",
    "BOOLEAN_OPERATORS",
    "WINDOW_TITLE",
    "WINDOW_SIZE",
    "PARQUET_PATH",
]


def main() -> None:
    """Entry point for the query builder UI."""
    root = tk.Tk()
    app = QueryBuilderApp(root)
    app.run()


if __name__ == "__main__":
    main()

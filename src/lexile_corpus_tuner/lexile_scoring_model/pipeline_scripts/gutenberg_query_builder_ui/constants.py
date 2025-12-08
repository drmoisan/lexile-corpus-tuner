"""Constants and configuration for Gutenberg Query Builder UI.

Defines field types, operators, window settings, and data paths.
"""

from __future__ import annotations

from pathlib import Path

# Window configuration
WINDOW_TITLE = "Gutenberg Query Builder"
WINDOW_SIZE = "1400x900"

# Data path (relative to repository root)
PARQUET_PATH = (
    Path(__file__).resolve().parents[5] / "data" / "meta" / "gutenberg_books.parquet"
)

# Field metadata: maps field names to their types
FIELD_TYPES = {
    "id": "numeric",
    "title": "text",
    "authors": "text",
    "subjects": "text",
    "bookshelves": "text",
    "languages": "text",
    "download_count": "numeric",
    "media_type": "text",
    "copyright": "boolean",
}

# Operators by field type
TEXT_OPERATORS = ["contains", "=", "!="]
NUMERIC_OPERATORS = [">", "<", ">=", "<=", "=", "!=", "range"]
BOOLEAN_OPERATORS = ["=", "!="]

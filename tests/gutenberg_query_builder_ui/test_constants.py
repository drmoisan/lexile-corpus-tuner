from __future__ import annotations

from pathlib import Path
from typing import Any


def test_constants_values(ui_modules: Any) -> None:
    constants = ui_modules.constants

    assert constants.WINDOW_TITLE == "Gutenberg Query Builder"
    assert constants.WINDOW_SIZE == "1400x900"
    assert isinstance(constants.PARQUET_PATH, Path)
    assert constants.FIELD_TYPES["id"] == "numeric"
    assert "contains" in constants.TEXT_OPERATORS
    assert ">" in constants.NUMERIC_OPERATORS
    assert "=" in constants.BOOLEAN_OPERATORS

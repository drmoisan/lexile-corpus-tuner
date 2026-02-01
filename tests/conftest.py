"""Global pytest configuration for repository-wide test helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Normalize legacy -k expressions that use pipe separators."""
    keyword_expr = config.option.keyword
    if not keyword_expr or "|" not in keyword_expr:
        return
    config.option.keyword = " or ".join(
        part.strip() for part in keyword_expr.split("|")
    )

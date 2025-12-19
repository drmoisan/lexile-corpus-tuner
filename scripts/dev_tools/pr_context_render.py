"""Compatibility wrapper for PR context rendering helpers."""

from __future__ import annotations

from typing import Any

from scripts.dev_tools.pr_context import render as _render

__all__ = [
    name for name in _render.__dict__ if not name.startswith("_")
]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    return getattr(_render, name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(_render.__dict__)))

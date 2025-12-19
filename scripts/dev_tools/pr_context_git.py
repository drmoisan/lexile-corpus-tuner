"""Compatibility wrapper for Git interaction helpers."""

from __future__ import annotations

from typing import Any

from scripts.dev_tools.pr_context import git as _git

__all__ = [
    name for name in _git.__dict__ if not name.startswith("_")
]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    return getattr(_git, name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(_git.__dict__)))

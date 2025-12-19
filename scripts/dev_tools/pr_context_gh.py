"""Compatibility wrapper for GitHub helpers."""

from __future__ import annotations

from typing import Any

from scripts.dev_tools.pr_context import github as _github

__all__ = [
    name for name in _github.__dict__ if not name.startswith("_")
]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    return getattr(_github, name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(_github.__dict__)))

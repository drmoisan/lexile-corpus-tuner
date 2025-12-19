"""Compatibility wrapper for PR context data models."""

from __future__ import annotations

from typing import Any

from scripts.dev_tools.pr_context import models as _models

__all__ = [
    name for name in _models.__dict__ if not name.startswith("_")
]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    return getattr(_models, name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(_models.__dict__)))

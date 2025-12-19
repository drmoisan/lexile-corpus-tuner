"""Compatibility wrapper for PR context collection entrypoints."""

from __future__ import annotations

from typing import Any

from scripts.dev_tools.pr_context import collector as _collector

__all__ = _collector.__all__  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    return getattr(_collector, name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(_collector.__dict__)))

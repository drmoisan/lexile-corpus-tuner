from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Governed JSON config globs (relative to repo root)
GOVERNED_GLOBS: Sequence[str] = (
    ".vscode/*.json",
    ".vscode/**/*.json",
    ".devcontainer/*.json",
    "scripts/**/*.json",
    "docs/**/*.json",
    "examples/**/*.json",
)

# Exclude large/generated data and artifacts
EXCLUDE_GLOBS: Sequence[str] = (
    "data/**",
    "artifacts/**",
    "htmlcov/**",
    "coverage*/**",
    "**/node_modules/**",
    ".venv",
    ".venv/**",
    "**/.venv",
    "**/.venv/**",
)


def iter_governed_files(root: Path | str) -> Iterable[Path]:
    """Yield governed JSON files under the repo root respecting excludes."""
    root_path = Path(root)
    # Resolve include matches first
    includes: list[Path] = []
    for pattern in GOVERNED_GLOBS:
        includes.extend(root_path.glob(pattern))

    # Build exclusion set
    excluded: set[Path] = set()
    for pattern in EXCLUDE_GLOBS:
        excluded.update(root_path.glob(pattern))

    for path in includes:
        if any(parent in excluded for parent in path.parents):
            continue
        if path in excluded:
            continue
        if path.is_file():
            yield path

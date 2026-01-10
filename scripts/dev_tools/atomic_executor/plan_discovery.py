"""Plan file discovery utilities for atomic executor.

Atomic executor historically assumed each feature folder contained `plan.md`.
This repository now also supports timestamped plan files of the form:

  plan.<YYYY-MM-DDTHH-mm>.md

This module centralizes plan-file selection so the CLI, prompt builder, and
feature resolver all agree on which plan file to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path  # noqa: TCH003 - Path required at runtime for I/O

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M"


@dataclass(frozen=True)
class ResolvedPlan:
    """Resolved plan file information.

    Purpose:
        Provide the on-disk plan path plus user-facing labels that keep prompts
        backward-compatible while still pointing to the correct file.

    Attributes:
        path (Path): The plan file path to read/write.
        display_label (str): Label to use in prompt section headings.
        update_filename (str): Filename users should update (checkbox flips).
    """

    path: Path
    display_label: str
    update_filename: str


def parse_timestamped_plan_filename(filename: str) -> datetime | None:
    """Parse a timestamped plan filename into a datetime.

    Purpose:
        Extract the timestamp from filenames like `plan.2026-01-09T22-27.md`.

    Args:
        filename (str): A filename (not a full path).

    Returns:
        datetime | None: Parsed timestamp if the name matches, otherwise None.
    """

    prefix = "plan."
    suffix = ".md"

    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None

    ts = filename[len(prefix) : -len(suffix)]

    try:
        return datetime.strptime(ts, _TIMESTAMP_FORMAT)
    except ValueError:
        return None


def resolve_feature_plan(feature_dir: Path) -> ResolvedPlan:
    """Resolve the plan file to use for a feature folder.

    Purpose:
        Prefer legacy `plan.md` when present. Otherwise, select the newest
        `plan.<timestamp>.md` file by timestamp.

    Args:
        feature_dir (Path): Feature folder containing plan/spec/story files.

    Returns:
        ResolvedPlan: Selected plan file details.

    Raises:
        FileNotFoundError: If no plan file is found.
    """

    legacy = feature_dir / "plan.md"
    if legacy.is_file():
        return ResolvedPlan(
            path=legacy, display_label="plan.md", update_filename="plan.md"
        )

    candidates = list(feature_dir.glob("plan.*.md"))

    newest: tuple[datetime, Path] | None = None

    # Select only files with well-formed timestamps, then pick the max.
    for candidate in candidates:
        ts = parse_timestamped_plan_filename(candidate.name)
        if ts is None:
            continue
        if newest is None or ts > newest[0]:
            newest = (ts, candidate)

    if newest is None:
        raise FileNotFoundError(
            f"Missing required plan file: {legacy} (or any plan.<timestamp>.md in "
            f"{feature_dir})"
        )

    chosen = newest[1]

    # Back-compat: keep the section markers as plan.md so older tests and
    # downstream tooling that grep for these markers continue to work.
    return ResolvedPlan(
        path=chosen,
        display_label="plan.md",
        update_filename=chosen.name,
    )


def is_plan_file_path(path: Path) -> bool:
    """Return True when the path looks like a supported plan file."""

    return (
        path.name == "plan.md" or parse_timestamped_plan_filename(path.name) is not None
    )

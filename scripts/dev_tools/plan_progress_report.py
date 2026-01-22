"""Generate a markdown report of incomplete plan checkboxes under docs/features/active.

Purpose:
    Provide a quick, deterministic snapshot of which active feature plans still have
    unchecked tasks. The report is intended to be written as a markdown table
    artifact (typically under artifacts/) and can be shared in PRs or issues.

Key rules (feature/type resolution):
    - We scan for plan files named exactly `plan.md` or matching `plan.*.md`.
    - A plan file's "feature" is normally the folder that contains it.
    - If the plan is inside a folder whose name starts with "v" (e.g., v1, v2),
      the feature is the *parent* folder of that v* folder.
    - The plan "type" is:
        - the v* folder name when inside a v* folder (e.g., "v2"), else
        - "base" when not in a v* folder.
    - We exclude plans with zero unchecked checkboxes.

Notes:
    This module intentionally separates pure parsing logic from filesystem I/O so
    that unit tests can avoid creating files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PlanProgressRow:
    """A single row of the plan progress report.

    Attributes:
        feature: The resolved feature folder name.
        issue: The issue reference, e.g. "#77", when present.
        plan_type: "base" or a version folder name like "v2".
        unchecked: Count of unchecked checkboxes.
        total: Total count of checkboxes.
    """

    feature: str
    issue: str
    plan_type: str
    unchecked: int
    total: int


_CHECKBOX_PATTERN = re.compile(r"^[ \t]*[-*]\s+\[(?P<state>[ xX])\]", re.MULTILINE)
_ISSUE_LINE_PATTERN = re.compile(
    r"^[ \t]*-\s*(?:\*\*Issue:\*\*|Issue:)\s*#(?P<num>\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TRAILING_ISSUE_IN_FEATURE_DIR_PATTERN = re.compile(r"-(?P<num>\d+)$")


def discover_plan_files(active_root: Path) -> list[Path]:
    """Discover plan markdown files under the active feature root.

    Purpose:
        Enumerate plan documents eligible for checkbox scanning.

    Args:
        active_root: Root directory (typically docs/features/active).

    Returns:
        A sorted list of plan file paths.
    """

    # Collect matching plan files using two glob patterns.
    # We intentionally keep the patterns narrow to avoid pulling in unrelated docs.
    plan_paths = (
        set(active_root.rglob("plan.md"))
        | set(active_root.rglob("plan.*.md"))
        | set(active_root.rglob("plan-*.md"))
    )
    return sorted(plan_paths)


def count_checkboxes(markdown: str) -> tuple[int, int]:
    """Count unchecked and total markdown task list checkboxes.

    Purpose:
        Identify remaining work at a glance by scanning `- [ ]` / `- [x]` lines.

    Args:
        markdown: Raw markdown text.

    Returns:
        A tuple of (unchecked, total).
    """

    unchecked = 0
    total = 0

    # Count all checkbox markers on list-item lines.
    for match in _CHECKBOX_PATTERN.finditer(markdown):
        total += 1
        if match.group("state") == " ":
            unchecked += 1

    return unchecked, total


def parse_primary_issue(markdown: str) -> str:
    """Extract the primary GitHub issue reference from a plan file.

    Purpose:
        Prefer the explicit "Issue: #NN" line in the plan header for accuracy.

    Args:
        markdown: Raw plan markdown.

    Returns:
        The issue reference (e.g., "#77") or an empty string when not found.
    """

    match = _ISSUE_LINE_PATTERN.search(markdown)
    if not match:
        return ""
    return f"#{match.group('num')}"


def resolve_feature_and_type(plan_path: Path, *, active_root: Path) -> tuple[str, str]:
    """Resolve feature name and plan type from the plan file path.

    Purpose:
        Apply the repo's folder convention for versioned plans.

    Args:
        plan_path: Path to a plan file under active_root.
        active_root: Root directory (docs/features/active).

    Returns:
        (feature, plan_type)

    Raises:
        ValueError: When plan_path is not located under active_root.
    """

    try:
        relative = plan_path.relative_to(active_root)
    except ValueError as exc:
        raise ValueError(f"plan_path is not under active_root: {plan_path}") from exc

    if len(relative.parts) < 2:
        # We expect at least <feature>/<plan-file>.
        return relative.parts[0] if relative.parts else plan_path.parent.name, "base"

    parent_dir_name = plan_path.parent.name
    if parent_dir_name.lower().startswith("v"):
        # Versioned subfolder: feature is the parent of the v* directory.
        feature_dir = plan_path.parent.parent
        return feature_dir.name, parent_dir_name

    return parent_dir_name, "base"


def fallback_issue_from_feature_dir(feature: str) -> str:
    """Infer an issue reference from a feature directory name.

    Purpose:
        Provide a reasonable fallback when a plan file lacks an explicit Issue line.

    Args:
        feature: Feature directory name (e.g., "2026-01-07-atomic-executor-77").

    Returns:
        The issue reference (e.g., "#77") or an empty string.
    """

    match = _TRAILING_ISSUE_IN_FEATURE_DIR_PATTERN.search(feature)
    if not match:
        return ""
    return f"#{match.group('num')}"


def build_report_rows(
    plan_docs: list[tuple[Path, str]], *, active_root: Path
) -> list[PlanProgressRow]:
    """Build report rows from plan docs.

    Purpose:
        Convert plan markdown documents into progress rows, excluding completed plans.

    Args:
        plan_docs: A list of (path, markdown_text).
        active_root: Root directory (docs/features/active).

    Returns:
        Rows for incomplete plans.
    """

    rows: list[PlanProgressRow] = []

    # Evaluate each plan independently to keep ordering stable and failures localized.
    for plan_path, text in plan_docs:
        feature, plan_type = resolve_feature_and_type(
            plan_path, active_root=active_root
        )
        unchecked, total = count_checkboxes(text)

        # Exclude plans with no tasks or no unchecked tasks.
        if total == 0 or unchecked == 0:
            continue

        issue = parse_primary_issue(text)
        if not issue:
            issue = fallback_issue_from_feature_dir(feature)

        rows.append(
            PlanProgressRow(
                feature=feature,
                issue=issue,
                plan_type=plan_type,
                unchecked=unchecked,
                total=total,
            )
        )

    # Sort for stable output: by feature name, then plan type.
    return sorted(rows, key=lambda r: (r.feature, r.plan_type))


def render_markdown_table(rows: list[PlanProgressRow]) -> str:
    """Render report rows as a markdown table.

    Args:
        rows: Report rows.

    Returns:
        Markdown content.
    """

    header = "| feature | issue | type | remaining |\n| --- | --- | --- | --- |"
    if not rows:
        return header + "\n"

    lines: list[str] = [header]

    # Format each row as unchecked/total for quick scanning.
    for row in rows:
        remaining = f"{row.unchecked}/{row.total}"
        lines.append(f"| {row.feature} | {row.issue} | {row.plan_type} | {remaining} |")

    return "\n".join(lines) + "\n"


def read_plan_docs(plan_paths: list[Path]) -> list[tuple[Path, str]]:
    """Read plan markdown documents from disk.

    Args:
        plan_paths: File paths to read.

    Returns:
        List of (path, content).
    """

    docs: list[tuple[Path, str]] = []

    # Read plans using UTF-8 to align with repo markdown conventions.
    for plan_path in plan_paths:
        docs.append((plan_path, plan_path.read_text(encoding="utf-8")))

    return docs


def generate_plan_progress_report(*, active_root: Path) -> str:
    """Generate the markdown report for incomplete plan checkboxes.

    Args:
        active_root: Root directory (docs/features/active).

    Returns:
        Markdown content for the report.
    """

    plan_paths = discover_plan_files(active_root)
    plan_docs = read_plan_docs(plan_paths)
    rows = build_report_rows(plan_docs, active_root=active_root)
    return render_markdown_table(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for report generation.

    Args:
        argv: Optional argv override for testing.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Scan docs/features/active for plan.md / plan.*.md and report "
            "incomplete checkboxes"
        )
    )
    parser.add_argument(
        "--active-root",
        type=Path,
        default=Path("docs/features/active"),
        help="Root folder to scan (default: docs/features/active)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/active_plan_progress.md"),
        help=(
            "Output markdown artifact path (default: "
            "artifacts/active_plan_progress.md)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Purpose:
        Generate a markdown table artifact showing incomplete plan tasks.

    Args:
        argv: Optional argv override for testing.

    Returns:
        Process exit code (0 for success).
    """

    args = _parse_args(argv)
    report = generate_plan_progress_report(active_root=args.active_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

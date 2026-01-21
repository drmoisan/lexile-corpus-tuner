"""Regression tests for bug template preservation in new_active_feature_folder.

Purpose:
    Ensure bug template structure (notably the `## Proposed Fix` subsections) is not
    clobbered by issue-seeded validation text when creating a new active bug folder.

    Also ensure we do *not* auto-fill the `### Design summary (what changes where):`
    subsection from the issue content. That subsection is intended to be filled later
    by an LLM using `issue.md` + research context.

Notes:
    These tests use an in-memory filesystem to comply with the repo policy that
    unit tests must not create temporary files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from scripts.dev_tools import new_active_feature_folder as mod

if TYPE_CHECKING:
    from collections.abc import Iterable


class FakeFileSystem(mod.FileSystem):
    """Minimal in-memory filesystem for deterministic unit tests."""

    def __init__(self) -> None:
        self.files: dict[Path, str] = {}
        self.dirs: set[Path] = set()

    def exists(self, path: Path) -> bool:
        return path in self.files or path in self.dirs

    def ensure_dir(self, path: Path) -> None:
        self.dirs.add(path)

    def copy_file(self, src: Path, dest: Path) -> None:
        if src not in self.files:
            raise FileNotFoundError(src)
        self.files[dest] = self.files[src]
        self.dirs.add(dest.parent)

    def copy_tree(self, src: Path, dest: Path) -> None:
        # Copy all file entries under src to dest preserving relative structure.
        for path, content in list(self.files.items()):
            try:
                relative = path.relative_to(src)
            except ValueError:
                continue
            self.files[dest / relative] = content
            self.dirs.add((dest / relative).parent)

    def list_files(self, path: Path) -> Iterable[Path]:
        return [file_path for file_path in self.files if file_path.parent == path]

    def read_text(self, path: Path) -> str:
        return self.files[path]

    def write_text(self, path: Path, content: str) -> None:
        self.files[path] = content
        self.dirs.add(path.parent)

    def move(self, src: Path, dest: Path) -> None:
        if src not in self.files:
            raise FileNotFoundError(src)
        self.files[dest] = self.files.pop(src)
        self.dirs.add(dest.parent)


def _seed_bug_template_matching_repo_template(
    fs: FakeFileSystem, workspace: Path
) -> None:
    """Seed a bug template that includes the important `## Proposed Fix` subsections."""

    template_dir = workspace / "docs" / "features" / "templates" / "bug"

    fs.write_text(
        template_dir / "spec.md",
        "\n".join(
            [
                "# <bug-name> (Spec)",
                "",
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "",
                "## Context",
                "",
                "## Repro & Evidence",
                "",
                "## Scope & Non-Goals",
                "",
                "## Root Cause Analysis",
                "",
                "## Proposed Fix",
                "",
                "### Design summary (what changes where):",
                "",
                "### Boundaries and invariants to preserve:",
                "",
                "### Dependencies or blocked work:",
                "",
                "### Implementation strategy (what changes, not sequencing):",
                "",
                "\t",  # preserve odd whitespace present in some templates
                "#### Files/modules to change:",
                "",
                "#### Functions/classes/CLI commands impacted:",
                "",
                "## Test Strategy",
                "- Regression tests to add or update:",
                "- Unit tests (pytest) for the fixed behavior and boundaries:",
                "",
                "## Acceptance Criteria",
                "- [ ] Repro steps now produce the expected behavior in all documented "
                "environments.",
            ]
        ),
    )

    fs.write_text(
        template_dir / "plan.yyyy-MM-ddTHH-mm.md",
        "\n".join(
            [
                "# <bug-name> (Plan)",
                "",
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
            ]
        ),
    )


def test_bug_folder_preserves_proposed_fix_template_subsections() -> None:
    """Regression: seeded validation must not delete `## Proposed Fix` structure."""

    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_bug_template_matching_repo_template(fs, workspace)

    # Seed a potential bug write-up that includes validation ideas.
    potential_dir = workspace / "docs" / "features" / "potential"
    potential_path = potential_dir / "atomic-executor-qc-regression-test.md"
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "## Summary",
                "bug summary",
                "## Proposed Fix / Validation Ideas",
                "validate with plan-linked expectations",
            ]
        ),
    )

    fixed_now = datetime(2026, 1, 21, 9, 57, tzinfo=ZoneInfo("America/New_York"))

    # Act
    result = mod.create_active_folder(
        feature_name="atomic-executor-qc-regression-test",
        feature_type="bug",
        issue_number="98",
        workspace=workspace,
        fs=fs,
        issue_fetcher=lambda _: None,
        code_launcher=lambda _: True,
        now_provider=lambda: fixed_now,
    )

    # Assert
    spec_path = result.target / "spec.md"
    spec_content = fs.read_text(spec_path)

    # Template structure should remain.
    assert "## Proposed Fix" in spec_content
    assert "### Design summary (what changes where):" in spec_content
    assert "### Boundaries and invariants to preserve:" in spec_content

    # The Design summary subsection should *not* be auto-filled from the issue.
    design_start = spec_content.index("### Design summary (what changes where):")
    design_end = spec_content.index(
        "### Boundaries and invariants to preserve:", design_start
    )
    design_block = spec_content[design_start:design_end]
    assert "validate with plan-linked expectations" not in design_block

    # Seeded validation text should still be present somewhere in the spec.
    assert "validate with plan-linked expectations" in spec_content

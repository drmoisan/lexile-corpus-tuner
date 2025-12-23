"""Tests for new_active_feature_folder Python implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.dev_tools import new_active_feature_folder as mod

if TYPE_CHECKING:
    from collections.abc import Iterable


class FakeFileSystem(mod.FileSystem):
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


def test_format_checklist_matches_expected_rules() -> None:
    raw = "Item one\n- [ ] existing\n- bullet\n   \nItem two"
    result = mod.format_checklist(raw)
    lines = result.splitlines()
    assert lines[0] == "- [ ] Item one"
    assert "- [ ] existing" in lines
    assert "- bullet" in lines
    assert lines[-1] == "- [ ] Item two"


def test_set_section_replaces_and_appends() -> None:
    content = "## Header\nold\n"
    updated = mod.set_section(content, "Header", "new")
    assert "new" in updated and "old" not in updated
    appended = mod.set_section(updated, "Another", "body")
    assert "## Another" in appended
    assert "body" in appended


def test_set_header_placeholder_replaces_placeholders() -> None:
    content = "- Owner: name\n- Last Updated: YYYY-MM-DD\n<feature-name> #<id>"
    result = mod.set_header_placeholder(
        content, "example", "#123", "owner", "2024-01-01"
    )
    assert "example" in result
    assert "#123" in result
    assert "owner" in result
    assert "2024-01-01" in result


def test_build_folder_slug_uses_potential_and_issue_number() -> None:
    potential = Path("/w/docs/features/potential/promoted/2025-12-23-json-quality.md")
    slug = mod.build_folder_slug("json-quality", potential, "63")
    assert slug == "2025-12-23-json-quality-63"


class FakeIssueFetcher:
    def __init__(self, meta: mod.IssueMeta | None = None) -> None:
        self.meta = meta
        self.calls: list[str] = []

    def __call__(self, issue_number: str) -> mod.IssueMeta | None:  # noqa: D401
        self.calls.append(issue_number)
        return self.meta


class FakeCodeLauncher:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    def __call__(self, files: Iterable[Path]) -> bool:  # noqa: D401
        file_list = list(files)
        self.calls.append(file_list)
        return True


def _seed_feature_template(fs: FakeFileSystem, workspace: Path) -> None:
    template_dir = workspace / "docs" / "features" / "templates" / "feature"
    fs.write_text(
        template_dir / "user-story.md",
        "\n".join(
            [
                "- Owner: name",
                "- Last Updated: YYYY-MM-DD",
                "<feature-name>",
                "## Problem / Why",
                "",
                "## Acceptance Criteria",
            ]
        ),
    )
    fs.write_text(
        template_dir / "spec.md",
        "\n".join(
            [
                "- Owner: name",
                "- Last Updated: YYYY-MM-DD",
                "<feature-name>",
                "## Overview",
                "",
                "## Behavior",
                "",
                "## Constraints & Risks",
                "",
                "## Seeded Test Conditions (from potential)",
            ]
        ),
    )
    fs.write_text(
        template_dir / "plan.md",
        "\n".join(
            [
                "- Owner: name",
                "- Last Updated: YYYY-MM-DD",
                "<feature-name>",
            ]
        ),
    )


def _seed_bug_template(fs: FakeFileSystem, workspace: Path) -> None:
    template_dir = workspace / "docs" / "features" / "templates" / "bug"
    fs.write_text(
        template_dir / "spec.md",
        "\n".join(
            [
                "- Owner: name",
                "- Last Updated: YYYY-MM-DD",
                "<feature-name>",
                "## Context",
                "## Repro & Evidence",
                "## Root Cause Analysis",
                "## Proposed Fix",
                "## Test Strategy",
            ]
        ),
    )
    fs.write_text(
        template_dir / "plan.md",
        "\n".join(
            [
                "- Owner: name",
                "- Last Updated: YYYY-MM-DD",
                "<feature-name>",
            ]
        ),
    )


def test_create_feature_folder_moves_potential_and_updates_files() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    potential_path = (
        workspace
        / "docs"
        / "features"
        / "potential"
        / "promoted"
        / "2025-12-23-json-quality.md"
    )
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "- Issue: #63",
                "## Problem / Why",
                "problem text",
                "## Proposed Behavior",
                "behavior text",
                "## Acceptance Criteria (early draft)",
                "first item",
                "## Constraints & Risks",
                "risk text",
                "## Test Conditions to Consider",
                "test A",
            ]
        ),
    )

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="json-quality",
        feature_type="feature",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    expected_folder = (
        workspace / "docs" / "features" / "active" / "2025-12-23-json-quality-63"
    )
    assert result.target == expected_folder
    assert result.potential_issue_path == expected_folder / "issue.md"
    assert potential_path not in fs.files
    assert fs.exists(expected_folder / "user-story.md")
    user_story = fs.read_text(expected_folder / "user-story.md")
    assert "problem text" in user_story
    assert "first item" in user_story
    assert "#63" in user_story
    assert code_launcher.calls, "code launcher should be invoked"


def test_create_bug_folder_uses_issue_metadata_and_sections() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_bug_template(fs, workspace)

    potential_path = workspace / "docs" / "features" / "potential" / "bug-case.md"
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "## Summary",
                "bug summary",
                "## Environment",
                "env",
                "## Steps to Reproduce",
                "step 1",
                "## Expected Behavior",
                "should work",
                "## Actual Behavior",
                "fails",
                "## Logs / Screenshots",
                "trace",
                "## Impact / Severity",
                "high",
                "## Suspected Cause / Notes",
                "root cause",
                "## Proposed Fix / Validation Ideas",
                "validate",
            ]
        ),
    )

    issue_meta = mod.IssueMeta(number="77", author="octocat", updated_date="2024-02-02")
    fetcher = FakeIssueFetcher(issue_meta)
    code_launcher = FakeCodeLauncher()

    result = mod.create_active_folder(
        feature_name="bug-case",
        feature_type="bug",
        issue_number="77",
        workspace=workspace,
        fs=fs,
        issue_fetcher=fetcher,
        code_launcher=code_launcher,
    )

    expected_folder = workspace / "docs" / "features" / "active" / "bug-case-77"
    assert result.target == expected_folder
    spec_content = fs.read_text(expected_folder / "spec.md")
    assert "bug summary" in spec_content
    assert "step 1" in spec_content
    assert "Expected:\nshould work" in spec_content
    assert "Actual:\nfails" in spec_content
    assert "Logs / Screenshots:\ntrace" in spec_content
    assert "Root Cause Analysis" in spec_content
    assert "Proposed Fix" in spec_content
    assert "#77" in spec_content
    assert "octocat" in spec_content
    assert "2024-02-02" in spec_content
    assert fetcher.calls == ["77"]
    assert code_launcher.calls, "code launcher should be invoked"


def test_validate_feature_name_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        mod.validate_feature_name("INVALID")

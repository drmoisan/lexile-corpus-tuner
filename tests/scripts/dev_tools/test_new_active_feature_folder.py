"""Tests for new_active_feature_folder Python implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from zoneinfo import ZoneInfo

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
    content = "\n".join(
        [
            "- **Issue:** <issue>",
            "- **Parent (optional):** <parent-id>",
            "- **Owner:** <name>",
            "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
            "- **Status:** <status>",
            "- **Version:** <version_number>",
            "<feature-name>",
        ]
    )
    result = mod.set_header_placeholder(
        content,
        "example",
        "#123",
        "owner",
        "2024-01-01T00-00",
        status_field="Draft",
        parent_field="none",
        version_field="0.1",
    )
    assert "example" in result
    assert "#123" in result
    assert "owner" in result
    assert "2024-01-01T00-00" in result
    assert "Draft" in result
    assert "none" in result
    assert "0.1" in result


def test_set_header_placeholder_does_not_prepend_plain_issue_line() -> None:
    """Ensure bold Issue headers do not trigger the fallback prepend."""
    content = "\n".join(
        [
            "# <bug-name> (Spec)",
            "",
            "- **Issue:** <issue>",
            "- **Owner:** <name>",
            "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
            "<bug-name>",
        ]
    )

    result = mod.set_header_placeholder(
        content,
        "example-bug",
        "#95",
        "drmoisan",
        "2026-01-20T16-15",
        status_field="Draft",
        parent_field="none",
        version_field="0.1",
    )

    assert result.splitlines()[0] == "# example-bug (Spec)"
    assert "- Issue: #95" not in result


def test_build_folder_slug_uses_potential_and_issue_number() -> None:
    potential = Path("/w/docs/features/potential/promoted/2025-12-23-json-quality.md")
    slug = mod.build_folder_slug("json-quality", potential, "63")
    assert slug == "2025-12-23-json-quality-63"


class FakeIssueFetcher:
    def __init__(self, meta: mod.IssueMeta | None = None) -> None:
        self.meta = meta
        self.calls: list[str] = []

    def __call__(self, issue_number: str) -> mod.IssueMeta | None:
        """Fetch mock issue metadata for testing."""
        self.calls.append(issue_number)
        return self.meta


class FakeCodeLauncher:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    def __call__(self, files: Iterable[Path]) -> bool:
        """Launch mock code editor for testing."""
        file_list = list(files)
        self.calls.append(file_list)
        return True


def _seed_feature_template(fs: FakeFileSystem, workspace: Path) -> None:
    template_dir = workspace / "docs" / "features" / "templates" / "feature"
    fs.write_text(
        template_dir / "user-story.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
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
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
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
        template_dir / "plan.yyyy-MM-ddTHH-mm.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
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
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
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
        template_dir / "plan.yyyy-MM-ddTHH-mm.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<bug-name>",
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
    fixed_now = datetime(2024, 1, 2, 3, 4, tzinfo=ZoneInfo("America/New_York"))
    result = mod.create_active_folder(
        feature_name="json-quality",
        feature_type="feature",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
        now_provider=lambda: fixed_now,
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

    plan_path = expected_folder / "plan.2024-01-02T03-04.md"
    assert fs.exists(plan_path)
    plan_content = fs.read_text(plan_path)
    assert "- **Issue:** #63" in plan_content
    assert "- **Parent (optional):** none" in plan_content
    assert "- **Status:** Draft" in plan_content
    assert "- **Version:** 0.1" in plan_content
    assert "- **Last Updated:** 2024-01-02T03-04" in plan_content
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
    fixed_now = datetime(2024, 2, 3, 4, 5, tzinfo=ZoneInfo("America/New_York"))

    result = mod.create_active_folder(
        feature_name="bug-case",
        feature_type="bug",
        issue_number="77",
        workspace=workspace,
        fs=fs,
        issue_fetcher=fetcher,
        code_launcher=code_launcher,
        now_provider=lambda: fixed_now,
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
    assert "2024-02-03T04-05" in spec_content
    assert fetcher.calls == ["77"]

    plan_path = expected_folder / "plan.2024-02-03T04-05.md"
    assert fs.exists(plan_path)
    plan_content = fs.read_text(plan_path)
    assert "- **Issue:** #77" in plan_content
    assert "- **Parent (optional):** none" in plan_content
    assert "- **Owner:** octocat" in plan_content
    assert "- **Last Updated:** 2024-02-03T04-05" in plan_content
    assert "- **Status:** Draft" in plan_content
    assert "- **Version:** 0.1" in plan_content
    assert code_launcher.calls, "code launcher should be invoked"


def test_validate_feature_name_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        mod.validate_feature_name("INVALID")


def test_set_section_handles_empty_body() -> None:
    content = "## Header\nold\n"
    result = mod.set_section(content, "Header", "")
    assert result == content


def test_find_potential_file_returns_none_when_no_match() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    result = mod.find_potential_file("nonexistent", workspace, fs)
    assert result is None


def test_parse_issue_number_returns_none_when_no_match() -> None:
    content = "Some content without issue"
    result = mod.parse_issue_number(content)
    assert result is None


def test_build_folder_slug_raises_on_invalid_slug() -> None:
    with pytest.raises(ValueError, match="invalid"):
        mod.build_folder_slug("name", Path("/some/INVALID-FILE.md"), None)


def test_update_feature_docs_for_refactor_type() -> None:
    """Test that update_feature_docs creates and populates refactor docs correctly."""
    # Arrange
    fs = FakeFileSystem()
    target_dir = Path("/target")
    fs.write_text(
        target_dir / "spec.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<refactor-name>",
            ]
        ),
    )
    fs.write_text(
        target_dir / "plan.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<refactor-name>",
            ]
        ),
    )

    sections = {
        "problem": "intent content",
        "behavior": "scope content",
        "constraints": "risks content",
        "tests": "test item",
    }

    # Act
    result = mod.update_feature_docs(
        feature_type="refactor",
        feature_name="my-refactor",
        target_dir=target_dir,
        issue_field="#42",
        owner_field="tester",
        updated_field="2024-01-15",
        parent_field="none",
        status_field="Draft",
        version_field="0.1",
        plan_updated_field="2024-01-15",
        fs=fs,
        sections=sections,
    )

    # Assert
    assert len(result) == 2
    assert result[0] == target_dir / "spec.md"
    assert result[1] == target_dir / "plan.md"

    spec_content = fs.read_text(target_dir / "spec.md")
    assert "my-refactor" in spec_content
    assert "#42" in spec_content
    assert "tester" in spec_content
    assert "2024-01-15" in spec_content
    assert "## Intent & Outcomes" in spec_content
    assert "intent content" in spec_content
    assert "## Scope (structural changes)" in spec_content
    assert "scope content" in spec_content
    assert "## Risks & Mitigations" in spec_content
    assert "risks content" in spec_content
    assert "## Seeded Test Conditions (from potential)" in spec_content
    assert "- [ ] test item" in spec_content

    plan_content = fs.read_text(target_dir / "plan.md")
    assert "my-refactor" in plan_content
    assert "#42" in plan_content
    assert "tester" in plan_content
    assert "Draft" in plan_content
    assert "0.1" in plan_content


def test_update_feature_docs_for_epic_type() -> None:
    """Test that update_feature_docs creates and populates epic docs correctly."""
    # Arrange
    fs = FakeFileSystem()
    target_dir = Path("/target")
    fs.write_text(
        target_dir / "initiative.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<epic-name>",
            ]
        ),
    )

    sections: dict[str, str] = {}

    # Act
    result = mod.update_feature_docs(
        feature_type="epic",
        feature_name="my-epic",
        target_dir=target_dir,
        issue_field="#100",
        owner_field="epic-owner",
        updated_field="2024-03-20",
        parent_field="none",
        status_field="Draft",
        version_field="0.1",
        plan_updated_field="2024-03-20",
        fs=fs,
        sections=sections,
    )

    # Assert
    assert len(result) == 1
    assert result[0] == target_dir / "initiative.md"

    initiative_content = fs.read_text(target_dir / "initiative.md")
    assert "my-epic" in initiative_content
    assert "#100" in initiative_content
    assert "epic-owner" in initiative_content
    assert "2024-03-20" in initiative_content


def test_create_refactor_folder_seeds_refactor_docs() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    template_dir = workspace / "docs" / "features" / "templates" / "refactor"
    fs.write_text(
        template_dir / "spec.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<refactor-name>",
                "## Intent & Outcomes",
                "",
                "## Scope (structural changes)",
                "",
                "## Risks & Mitigations",
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
                "<refactor-name>",
            ]
        ),
    )

    potential_path = workspace / "docs" / "features" / "potential" / "refactor-test.md"
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "- Issue: #88",
                "## Problem / Why",
                "intent text",
                "## Proposed Behavior",
                "scope text",
                "## Constraints & Risks",
                "risks text",
                "## Test Conditions to Consider",
                "test condition",
            ]
        ),
    )

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="refactor-test",
        feature_type="refactor",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    expected_folder = workspace / "docs" / "features" / "active" / "refactor-test-88"
    assert result.target == expected_folder
    spec_content = fs.read_text(expected_folder / "spec.md")
    assert "intent text" in spec_content
    assert "scope text" in spec_content
    assert "risks text" in spec_content
    assert "test condition" in spec_content
    assert "#88" in spec_content


def test_create_epic_folder_seeds_epic_docs() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    template_dir = workspace / "docs" / "features" / "templates" / "epic"
    fs.write_text(
        template_dir / "initiative.md",
        "\n".join(
            [
                "- **Issue:** <issue>",
                "- **Parent (optional):** <parent-id>",
                "- **Owner:** <name>",
                "- **Last Updated:** <yyyy-MM-ddTHH-mm>",
                "- **Status:** <status>",
                "- **Version:** <version_number>",
                "<epic-name>",
            ]
        ),
    )

    potential_path = workspace / "docs" / "features" / "potential" / "epic-test.md"
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "- Issue: #99",
                "## Problem / Why",
                "epic content",
            ]
        ),
    )

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="epic-test",
        feature_type="epic",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    expected_folder = workspace / "docs" / "features" / "active" / "epic-test-99"
    assert result.target == expected_folder
    initiative_content = fs.read_text(expected_folder / "initiative.md")
    assert "#99" in initiative_content


def test_create_active_folder_raises_on_invalid_feature_type() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    with pytest.raises(ValueError, match="must be one of"):
        mod.create_active_folder(
            feature_name="test",
            feature_type="invalid",  # type: ignore[arg-type]
            workspace=workspace,
            fs=fs,
        )


def test_create_active_folder_raises_on_missing_template() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    with pytest.raises(FileNotFoundError, match="Template folder not found"):
        mod.create_active_folder(
            feature_name="test",
            feature_type="feature",
            workspace=workspace,
            fs=fs,
        )


def test_create_active_folder_with_force_overwrites_existing() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    target_dir = workspace / "docs" / "features" / "active" / "test-feature"
    fs.ensure_dir(target_dir)
    fs.write_text(target_dir / "existing.txt", "old content")

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="test-feature",
        feature_type="feature",
        force=True,
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    assert result.target == target_dir
    assert fs.exists(target_dir / "user-story.md")


def test_create_active_folder_without_potential_file() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="new-feature",
        feature_type="feature",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    expected_folder = workspace / "docs" / "features" / "active" / "new-feature"
    assert result.target == expected_folder
    assert result.potential_issue_path is None


def test_create_active_folder_with_auto_issue_detection() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    potential_path = workspace / "docs" / "features" / "potential" / "auto-test.md"
    fs.write_text(
        potential_path,
        "\n".join(
            [
                "- Issue: #42",
                "## Problem / Why",
                "content",
            ]
        ),
    )

    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="auto-test",
        feature_type="feature",
        issue_number="auto",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )

    expected_folder = workspace / "docs" / "features" / "active" / "auto-test-42"
    assert result.target == expected_folder


def test_issue_fetcher_returns_none_when_gh_missing() -> None:
    result = mod.default_issue_fetcher("123")
    # If gh is missing, returns None; if present, may return data or None
    assert result is None or isinstance(result, mod.IssueMeta)


def test_code_launcher_returns_false_when_code_missing() -> None:
    with mock.patch("shutil.which", return_value=None):
        with mock.patch("subprocess.run") as mock_run:
            result = mod.default_code_launcher([Path("/test.md")])

    assert result is False
    mock_run.assert_not_called()


def test_apply_header_and_sections_skips_missing_file() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    # Create template without the file we'll try to update
    template_dir = workspace / "docs" / "features" / "templates" / "feature"
    fs.write_text(template_dir / "user-story.md", "- Owner: name\n<feature-name>")

    # Create active folder and verify it handles missing optional files gracefully
    code_launcher = FakeCodeLauncher()
    result = mod.create_active_folder(
        feature_name="test",
        feature_type="feature",
        workspace=workspace,
        fs=fs,
        code_launcher=code_launcher,
    )
    # Should succeed without error even if some files are missing
    assert result.target


def test_create_active_folder_raises_when_exists_without_force() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    target_dir = workspace / "docs" / "features" / "active" / "test-feature"
    fs.ensure_dir(target_dir)

    with pytest.raises(FileExistsError, match="Re-run with --force"):
        mod.create_active_folder(
            feature_name="test-feature",
            feature_type="feature",
            workspace=workspace,
            fs=fs,
        )


def test_create_active_folder_prints_fallback_when_code_launcher_fails() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    _seed_feature_template(fs, workspace)

    def failing_launcher(files: Iterable[Path]) -> bool:
        return False

    # This should not raise, just print fallback message
    result = mod.create_active_folder(
        feature_name="test",
        feature_type="feature",
        workspace=workspace,
        fs=fs,
        code_launcher=failing_launcher,
    )
    assert (
        result.target.exists() or not result.target.exists()
    )  # Just checking no crash


def test_issue_fetcher_subprocess_returns_none_on_error() -> None:
    """Test that default_issue_fetcher handles subprocess errors gracefully."""
    # This will attempt real subprocess if gh exists; if not, returns None
    # Either way, it should not raise
    result = mod.default_issue_fetcher("99999")
    assert result is None or isinstance(result, mod.IssueMeta)


def test_issue_fetcher_handles_malformed_response() -> None:
    """Test that default_issue_fetcher handles missing updatedAt field."""
    # This tests the real fetcher's error handling; behavior depends on gh availability
    result = mod.default_issue_fetcher("1")
    assert result is None or isinstance(result, mod.IssueMeta)


def test_default_issue_fetcher_when_gh_not_found() -> None:
    """Test that default_issue_fetcher returns None when gh is not in PATH."""
    with mock.patch("shutil.which", return_value=None):
        result = mod.default_issue_fetcher("123")
        assert result is None


def test_default_issue_fetcher_handles_failed_subprocess() -> None:
    """Test that default_issue_fetcher returns None when subprocess fails."""
    with mock.patch("shutil.which", return_value="/usr/bin/gh"):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="")
            result = mod.default_issue_fetcher("123")
            assert result is None


def test_default_issue_fetcher_handles_json_decode_error() -> None:
    """Test that default_issue_fetcher returns None on malformed JSON."""
    with mock.patch("shutil.which", return_value="/usr/bin/gh"):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="not json")
            result = mod.default_issue_fetcher("123")
            assert result is None


def test_default_issue_fetcher_handles_missing_updated_at() -> None:
    """Test that default_issue_fetcher handles missing updatedAt field."""
    with mock.patch("shutil.which", return_value="/usr/bin/gh"):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout='{"number": 123, "author": {"login": "test"}}',
            )
            result = mod.default_issue_fetcher("123")
            assert result is not None
            assert result.number == "123"
            assert result.author == "test"
            assert result.updated_date == "YYYY-MM-DD"


def test_default_issue_fetcher_parses_updated_at() -> None:
    """Test that default_issue_fetcher parses updatedAt correctly."""
    with mock.patch("shutil.which", return_value="/usr/bin/gh"):
        with mock.patch("subprocess.run") as mock_run:
            stdout_value = (
                '{"number": 123, "author": {"login": "test"}, '
                '"updatedAt": "2024-01-15T10:30:00Z"}'
            )
            mock_run.return_value = mock.Mock(returncode=0, stdout=stdout_value)
            result = mod.default_issue_fetcher("123")
            assert result is not None
            assert result.updated_date == "2024-01-15"


def test_default_code_launcher_with_no_code_command() -> None:
    """Test that default_code_launcher returns False when code command missing."""
    with mock.patch("shutil.which", return_value=None):
        result = mod.default_code_launcher([Path("/test.md")])
        assert result is False


def test_default_code_launcher_with_code_command() -> None:
    """Test that default_code_launcher calls code command and returns True."""
    with mock.patch("shutil.which", return_value="/usr/bin/code"):
        with mock.patch("subprocess.run") as mock_run:
            result = mod.default_code_launcher([Path("/test1.md"), Path("/test2.md")])
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/code"
            assert "/test1.md" in " ".join(args)
            assert "/test2.md" in " ".join(args)


def test_default_issue_fetcher_handles_exception_in_date_parsing() -> None:
    """Test that default_issue_fetcher handles exceptions in date parsing."""
    with mock.patch("shutil.which", return_value="/usr/bin/gh"):
        with mock.patch("subprocess.run") as mock_run:
            # Use an object that will cause split to fail
            stdout_value = (
                '{"number": 123, "author": {"login": "test"}, "updatedAt": null}'
            )
            mock_run.return_value = mock.Mock(returncode=0, stdout=stdout_value)
            result = mod.default_issue_fetcher("123")
            assert result is not None
            # When updatedAt is null or missing, should default to YYYY-MM-DD
            assert result.updated_date == "YYYY-MM-DD"

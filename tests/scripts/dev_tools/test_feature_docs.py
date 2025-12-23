"""Unit tests for scripts/dev_tools/pr_context/feature_docs.py module."""

from pathlib import Path

from scripts.dev_tools.pr_context.feature_docs import (
    _resolve_feature_dir,  # pyright: ignore[reportPrivateUsage]
    completed_plan_tasks,
    extract_issue_references,
    gather_feature_excerpts,
    parse_section,
)


class TestParseSection:
    def test_parse_section_found(self) -> None:
        markdown = "## Introduction\nHello\n## Details\nMore info\n## End\nFinal"
        result = parse_section(markdown, "Details")
        assert result == "More info"

    def test_parse_section_not_found(self) -> None:
        markdown = "## Introduction\nContent"
        result = parse_section(markdown, "Missing")
        assert result == ""

    def test_parse_section_last_heading(self) -> None:
        markdown = "## First\nOne\n## Last\nTwo"
        result = parse_section(markdown, "Last")
        assert result == "Two"

    def test_parse_section_empty_content(self) -> None:
        markdown = "## Empty\n## Next\nContent"
        result = parse_section(markdown, "Empty")
        assert result == ""

    def test_parse_section_special_chars(self) -> None:
        markdown = "## Details (v2.0)\nContent here"
        result = parse_section(markdown, "Details (v2.0)")
        assert result == "Content here"


class TestCompletedPlanTasks:
    def test_completed_plan_tasks_lowercase_x(self) -> None:
        markdown = "- [x] Task 1\n- [ ] Task 2\n- [x] Task 3"
        result = completed_plan_tasks(markdown)
        assert result == ["Task 1", "Task 3"]

    def test_completed_plan_tasks_uppercase_x(self) -> None:
        markdown = "- [X] Done\n- [ ] Todo"
        result = completed_plan_tasks(markdown)
        assert result == ["Done"]

    def test_completed_plan_tasks_limit(self) -> None:
        markdown = "- [x] A\n- [x] B\n- [x] C"
        result = completed_plan_tasks(markdown, limit=2)
        assert result == ["A", "B"]

    def test_completed_plan_tasks_asterisk_bullets(self) -> None:
        markdown = "* [x] Task A\n* [ ] Task B"
        result = completed_plan_tasks(markdown)
        assert result == ["Task A"]

    def test_completed_plan_tasks_no_completed(self) -> None:
        markdown = "- [ ] Todo 1\n- [ ] Todo 2"
        result = completed_plan_tasks(markdown)
        assert result == []


class TestExtractIssueReferences:
    def test_extract_issue_references_github(self) -> None:
        text = "Relates to #123 and #456"
        result = extract_issue_references(text)
        assert result == ["#123", "#456"]

    def test_extract_issue_references_jira(self) -> None:
        text = "See ABC-123 and XYZ-456"
        result = extract_issue_references(text)
        assert result == ["ABC-123", "XYZ-456"]

    def test_extract_issue_references_mixed(self) -> None:
        text = "Fix #42 for PROJECT-100"
        result = extract_issue_references(text)
        assert result == ["#42", "PROJECT-100"]

    def test_extract_issue_references_deduplication(self) -> None:
        text = "#10 again #10 and #10"
        result = extract_issue_references(text)
        assert result == ["#10"]

    def test_extract_issue_references_empty(self) -> None:
        result = extract_issue_references("")
        assert result == []

    def test_extract_issue_references_no_matches(self) -> None:
        text = "Just plain text"
        result = extract_issue_references(text)
        assert result == []


class TestGatherFeatureExcerpts:
    def test_gather_feature_excerpts_direct_match(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "docs" / "features" / "active" / "test-feature"
        feature_dir.mkdir(parents=True)
        (tmp_path / "docs" / "features" / "potential" / "promoted").mkdir(parents=True)

        user_story = feature_dir / "user-story.md"
        user_story.write_text(
            "## Problem / Why\nAs a user I need this feature...", encoding="utf-8"
        )

        spec = feature_dir / "spec.md"
        spec.write_text(
            "## Overview\nFeature spec.\n## Details\nMore info.", encoding="utf-8"
        )

        plan = feature_dir / "plan.md"
        plan.write_text("## Tasks\n- [x] Task 1\n- [ ] Task 2", encoding="utf-8")

        changed_files = ["docs/features/active/test-feature/user-story.md"]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)

        assert len(excerpts) == 1
        assert excerpts[0].feature == "test-feature"
        assert "As a user I need this feature..." in excerpts[0].excerpt
        assert "Feature spec." in excerpts[0].excerpt
        assert "Task 1" in excerpts[0].excerpt

    def test_gather_feature_excerpts_fuzzy_match(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-test-feature-impl"
        feature_dir.mkdir(parents=True)
        (tmp_path / "docs" / "features" / "potential" / "promoted").mkdir(parents=True)

        spec = feature_dir / "spec.md"
        spec.write_text("## Spec\nData", encoding="utf-8")

        changed_files = ["docs/features/active/my-test-feature-impl/spec.md"]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)

        assert len(excerpts) == 1

    def test_gather_feature_excerpts_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "docs" / "features" / "active").mkdir(parents=True)
        (tmp_path / "docs" / "features" / "potential" / "promoted").mkdir(parents=True)

        changed_files = ["docs/features/active/nonexistent/user-story.md"]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)

        assert len(excerpts) == 0

    def test_gather_feature_excerpts_promoted(self, tmp_path: Path) -> None:
        (tmp_path / "docs" / "features" / "active").mkdir(parents=True)

        promoted_dir = tmp_path / "docs" / "features" / "potential" / "promoted"
        feature_dir = promoted_dir / "test-feature"
        feature_dir.mkdir(parents=True)

        user_story = feature_dir / "user-story.md"
        user_story.write_text(
            "## Problem / Why\nPromoted story content", encoding="utf-8"
        )

        changed_files = ["docs/features/active/test-feature/plan.md"]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)
        assert len(excerpts) == 1
        assert excerpts[0].feature == "test-feature"
        assert "Promoted story content" in excerpts[0].excerpt

    def test_gather_feature_excerpts_extracts_issue_refs(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "docs" / "features" / "active" / "test"
        feature_dir.mkdir(parents=True)
        (tmp_path / "docs" / "features" / "potential" / "promoted").mkdir(parents=True)

        user_story = feature_dir / "user-story.md"
        user_story.write_text("Relates to #123 and ABC-456", encoding="utf-8")

        changed_files = ["docs/features/active/test/user-story.md"]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)

        assert len(excerpts) == 1
        assert "#123" in excerpts[0].issue_refs
        assert "ABC-456" in excerpts[0].issue_refs

    def test_gather_feature_excerpts_multiple_features(self, tmp_path: Path) -> None:
        for name in ["feature-a", "feature-b"]:
            feature_dir = tmp_path / "docs" / "features" / "active" / name
            feature_dir.mkdir(parents=True)
            (feature_dir / "user-story.md").write_text(
                f"Story for {name}", encoding="utf-8"
            )
        (tmp_path / "docs" / "features" / "potential" / "promoted").mkdir(parents=True)

        changed_files = [
            "docs/features/active/feature-a/user-story.md",
            "docs/features/active/feature-b/spec.md",
        ]
        excerpts = gather_feature_excerpts(tmp_path, changed_files)

        assert len(excerpts) == 2
        features = {e.feature for e in excerpts}
        assert features == {"feature-a", "feature-b"}


class TestResolveFeatureDir:
    """Tests for _resolve_feature_dir focusing on directory matching loop."""

    def test_resolve_feature_dir_direct_match(self, tmp_path: Path) -> None:
        """Test direct match when feature folder exists exactly."""
        base_dir = tmp_path / "active"
        feature_dir = base_dir / "my-feature"
        feature_dir.mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result == feature_dir

    def test_resolve_feature_dir_pattern_match_prefix(self, tmp_path: Path) -> None:
        """Test pattern matching with feature at start of directory name."""
        base_dir = tmp_path / "active"
        (base_dir / "2025-12-01-my-feature-impl").mkdir(parents=True)
        (base_dir / "other-folder").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        assert result.name == "2025-12-01-my-feature-impl"

    def test_resolve_feature_dir_pattern_match_suffix(self, tmp_path: Path) -> None:
        """Test pattern matching with feature at end of directory name."""
        base_dir = tmp_path / "active"
        (base_dir / "impl-my-feature").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        assert result.name == "impl-my-feature"

    def test_resolve_feature_dir_pattern_match_middle(self, tmp_path: Path) -> None:
        """Test pattern matching with feature in middle of directory name."""
        base_dir = tmp_path / "active"
        (base_dir / "prefix-my-feature-suffix").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        assert result.name == "prefix-my-feature-suffix"

    def test_resolve_feature_dir_weak_match(self, tmp_path: Path) -> None:
        """Test weak substring match when no pattern match found."""
        base_dir = tmp_path / "active"
        (base_dir / "somemyfeaturedir").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "myfeature")
        assert result is not None
        assert result.name == "somemyfeaturedir"

    def test_resolve_feature_dir_strong_over_weak(self, tmp_path: Path) -> None:
        """Test that strong pattern match is preferred over weak substring match."""
        base_dir = tmp_path / "active"
        (base_dir / "weak-myfeature-match").mkdir(parents=True)
        (base_dir / "strong-my-feature-match").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        # Strong match (with delimiters) should win
        assert result.name == "strong-my-feature-match"

    def test_resolve_feature_dir_skips_files(self, tmp_path: Path) -> None:
        """Test that files are skipped during directory iteration."""
        base_dir = tmp_path / "active"
        base_dir.mkdir(parents=True)
        # Create a file (not directory) with matching name
        (base_dir / "my-feature.txt").write_text("not a dir", encoding="utf-8")
        # Create actual directory
        (base_dir / "my-feature-dir").mkdir()

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        assert result.name == "my-feature-dir"

    def test_resolve_feature_dir_sorted_order(self, tmp_path: Path) -> None:
        """Test first sorted match returned when multiple strong matches exist."""
        base_dir = tmp_path / "active"
        (base_dir / "z-my-feature").mkdir(parents=True)
        (base_dir / "a-my-feature").mkdir(parents=True)
        (base_dir / "m-my-feature").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "my-feature")
        assert result is not None
        # Should return first in sorted order
        assert result.name == "a-my-feature"

    def test_resolve_feature_dir_no_match(self, tmp_path: Path) -> None:
        """Test returns None when no match found."""
        base_dir = tmp_path / "active"
        (base_dir / "other-feature").mkdir(parents=True)
        (base_dir / "different-thing").mkdir(parents=True)

        result = _resolve_feature_dir(base_dir, "nonexistent")
        assert result is None

    def test_resolve_feature_dir_empty_directory(self, tmp_path: Path) -> None:
        """Test returns None when base directory is empty."""
        base_dir = tmp_path / "active"
        base_dir.mkdir()

        result = _resolve_feature_dir(base_dir, "any-feature")
        assert result is None

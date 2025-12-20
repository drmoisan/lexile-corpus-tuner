"""Unit tests for refactored helper functions in render.py."""

from pathlib import Path

from scripts.dev_tools.pr_context.render import (
    build_excerpt_text,
    extract_features_from_paths,
    extract_plan_sections,
    extract_spec_parts,
    extract_story_parts,
    gather_feature_excerpts,
    read_text_file,
    resolve_feature_dir,
)


class TestResolveFeatureDir:
    def test_exact_match(self, tmp_path: Path) -> None:
        """resolve_feature_dir returns directory with exact name match."""
        base = tmp_path / "features"
        feature_dir = base / "my-feature"
        feature_dir.mkdir(parents=True)
        result = resolve_feature_dir(base, "my-feature")
        assert result == feature_dir

    def test_strong_pattern_match(self, tmp_path: Path) -> None:
        """resolve_feature_dir matches feature names with delimiters."""
        base = tmp_path / "features"
        feature_dir = base / "prefix-my-feature-suffix"
        feature_dir.mkdir(parents=True)
        result = resolve_feature_dir(base, "my-feature")
        assert result == feature_dir

    def test_weak_substring_match(self, tmp_path: Path) -> None:
        """resolve_feature_dir falls back to substring match."""
        base = tmp_path / "features"
        feature_dir = base / "somemyfeaturename"
        feature_dir.mkdir(parents=True)
        result = resolve_feature_dir(base, "myfeature")
        assert result == feature_dir

    def test_no_match(self, tmp_path: Path) -> None:
        """resolve_feature_dir returns None when no match found."""
        base = tmp_path / "features"
        base.mkdir()
        result = resolve_feature_dir(base, "nonexistent")
        assert result is None

    def test_missing_base_dir(self, tmp_path: Path) -> None:
        """resolve_feature_dir returns None when base doesn't exist."""
        result = resolve_feature_dir(tmp_path / "missing", "feature")
        assert result is None

    def test_ignores_files_in_fuzzy_search(self, tmp_path: Path) -> None:
        """resolve_feature_dir ignores files during pattern/fuzzy matching."""
        base = tmp_path / "features"
        base.mkdir()
        (base / "myfeature-file").write_text("file not dir")
        result = resolve_feature_dir(base, "myfeature")
        assert result is None


class TestReadTextFile:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        """read_text_file reads content from existing file."""
        file = tmp_path / "test.txt"
        file.write_text("test content", encoding="utf-8")
        result = read_text_file(file)
        assert result == "test content"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """read_text_file returns empty string for missing file."""
        result = read_text_file(tmp_path / "missing.txt")
        assert result == ""


class TestExtractFeaturesFromPaths:
    def test_extracts_from_active_paths(self) -> None:
        """extract_features_from_paths gets names from docs/features/active/**."""
        paths = [
            "docs/features/active/feature-a/spec.md",
            "docs/features/active/feature-b/plan.md",
            "docs/features/active/feature-a/user-story.md",
        ]
        result = extract_features_from_paths(paths)
        assert result == {"feature-a", "feature-b"}

    def test_ignores_non_active_paths(self) -> None:
        """extract_features_from_paths only processes active paths."""
        paths = [
            "docs/features/potential/feature-a/spec.md",
            "README.md",
            "src/module.py",
        ]
        result = extract_features_from_paths(paths)
        assert result == set()

    def test_requires_minimum_depth(self) -> None:
        """extract_features_from_paths requires at least 4 path parts."""
        paths = ["docs/features/active"]
        result = extract_features_from_paths(paths)
        assert result == set()

    def test_empty_list(self) -> None:
        """extract_features_from_paths handles empty list."""
        result = extract_features_from_paths([])
        assert result == set()


class TestExtractSpecParts:
    def test_finds_known_headings(self) -> None:
        """extract_spec_parts extracts recognized section headings."""
        spec = "## Context\nCtx\n## Problem\nProb\n## Unknown\nIgnored"
        result = extract_spec_parts(spec)
        assert len(result) == 2
        assert any("Context:" in part for part in result)
        assert any("Problem:" in part for part in result)
        assert not any("Unknown:" in part for part in result)

    def test_no_known_headings(self) -> None:
        """extract_spec_parts returns empty list when no recognized headings."""
        spec = "## Unknown\nContent"
        result = extract_spec_parts(spec)
        assert result == []

    def test_empty_spec(self) -> None:
        """extract_spec_parts handles empty spec text."""
        result = extract_spec_parts("")
        assert result == []


class TestExtractPlanSections:
    def test_extracts_tasks_and_test_plan(self) -> None:
        """extract_plan_sections gets completed tasks and test plan."""
        plan = "- [x] Task 1\n- [ ] Task 2\n- [x] Task 3\n## Test Plan\nDetails"
        plan_section, verification = extract_plan_sections(plan)
        assert "Task 1" in plan_section
        assert "Task 3" in plan_section
        assert "Task 2" not in plan_section
        assert "Details" in verification

    def test_no_completed_tasks(self) -> None:
        """extract_plan_sections returns empty when no completed tasks."""
        plan = "- [ ] Task 1\n- [ ] Task 2"
        plan_section, _ = extract_plan_sections(plan)
        assert plan_section == ""

    def test_no_test_plan_section(self) -> None:
        """extract_plan_sections returns empty verification when no test plan."""
        plan = "- [x] Task 1"
        _, verification = extract_plan_sections(plan)
        assert verification == ""


class TestExtractStoryParts:
    def test_extracts_statement_and_problem(self) -> None:
        """extract_story_parts gets story statement and problem sections."""
        story = "## Story Statement\n- As user\n- I want\n## Problem / Why\nBecause"
        result = extract_story_parts(story, "")
        assert len(result) == 2
        assert "Story Statement:" in result[0]
        assert "Problem / Why:" in result[1]

    def test_falls_back_to_promoted(self) -> None:
        """extract_story_parts uses promoted story when primary is empty."""
        promoted = "## Problem / Why\nPromoted problem"
        result = extract_story_parts("", promoted)
        assert len(result) == 1
        assert "Promoted problem" in result[0]

    def test_promoted_summary_fallback(self) -> None:
        """extract_story_parts tries Summary if Problem/Why not in promoted."""
        promoted = "## Summary\nSummary content"
        result = extract_story_parts("", promoted)
        assert len(result) == 1
        assert "Summary content" in result[0]

    def test_empty_stories(self) -> None:
        """extract_story_parts returns empty list when no content."""
        result = extract_story_parts("", "")
        assert result == []


class TestBuildExcerptText:
    def test_includes_all_sections(self) -> None:
        """build_excerpt_text combines all provided sections."""
        result = build_excerpt_text(
            "feat",
            ["Story part"],
            ["Spec part"],
            "Plan part",
            "Verification part",
        )
        assert "Feature doc: feat" in result
        assert "Story part" in result
        assert "Spec part" in result
        assert "Plan part" in result
        assert "Verification part" in result

    def test_omits_empty_sections(self) -> None:
        """build_excerpt_text skips empty sections."""
        result = build_excerpt_text("feat", [], ["Spec"], "", "")
        assert "Spec" in result
        assert "Story" not in result
        assert "Plan" not in result

    def test_shows_placeholder_when_empty(self) -> None:
        """build_excerpt_text shows placeholder when no content."""
        result = build_excerpt_text("feat", [], [], "", "")
        assert "(no spec/plan/user-story excerpts found)" in result


class TestGatherFeatureExcerptsIntegration:
    def test_full_integration_with_all_docs(self, tmp_path: Path) -> None:
        """gather_feature_excerpts integrates all helpers."""
        active = tmp_path / "docs" / "features" / "active"
        feature_dir = active / "test-feature"
        feature_dir.mkdir(parents=True)

        (feature_dir / "spec.md").write_text("## Context\nCtx\n## Problem\nProb")
        (feature_dir / "plan.md").write_text("- [x] Done\n## Test Plan\nTested")
        (feature_dir / "user-story.md").write_text(
            "## Story Statement\n- As user\n## Problem / Why\nNeed"
        )

        changed = ["docs/features/active/test-feature/spec.md"]
        result = gather_feature_excerpts(tmp_path, changed)

        assert len(result) == 1
        assert result[0].feature == "test-feature"
        assert "Context:" in result[0].excerpt
        assert "Done" in result[0].excerpt

    def test_multiple_features(self, tmp_path: Path) -> None:
        """gather_feature_excerpts processes multiple features."""
        active = tmp_path / "docs" / "features" / "active"
        for name in ["feat-a", "feat-b"]:
            feature_dir = active / name
            feature_dir.mkdir(parents=True)
            (feature_dir / "spec.md").write_text("## Overview\nContent")

        changed = [
            "docs/features/active/feat-a/spec.md",
            "docs/features/active/feat-b/plan.md",
        ]
        result = gather_feature_excerpts(tmp_path, changed)

        assert len(result) == 2
        assert {r.feature for r in result} == {"feat-a", "feat-b"}

    def test_missing_feature_directory_skipped(self, tmp_path: Path) -> None:
        """gather_feature_excerpts skips features with no directory."""
        changed = ["docs/features/active/nonexistent/spec.md"]
        result = gather_feature_excerpts(tmp_path, changed)
        assert result == []

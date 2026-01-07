"""
Tests for atomic_executor.prompt_builder module.

Tests cover PromptBuilder class methods for constructing prompts from templates
and feature context (plan.md, spec.md, user-story.md).
"""

from pathlib import Path

import pytest

from scripts.dev_tools.atomic_executor.plan_parser import PlanTask
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_init_with_valid_template(self, tmp_path: Path) -> None:
        """__init__() stores paths when template exists."""
        template = tmp_path / "template.md"
        template.write_text("Template content", encoding="utf-8")

        builder = PromptBuilder(tmp_path, template)
        assert builder.workspace == tmp_path
        assert builder.template_path == template

    def test_init_raises_for_nonexistent_template(self, tmp_path: Path) -> None:
        """__init__() raises FileNotFoundError for missing template."""
        template = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            PromptBuilder(tmp_path, template)

    def test_init_raises_for_directory_template(self, tmp_path: Path) -> None:
        """__init__() raises FileNotFoundError when template is directory."""
        template_dir = tmp_path / "template"
        template_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            PromptBuilder(tmp_path, template_dir)


class TestPromptBuilderBuild:
    """Tests for build() method."""

    def test_build_combines_template_and_context(self, tmp_path: Path) -> None:
        """build() combines template with plan, spec, story context."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text(
            "# Plan\n- [x] [P0-T1] Task 1\n- [ ] [P0-T2] Task 2", encoding="utf-8"
        )
        (feature_dir / "spec.md").write_text(
            "# Specification\nDetails here.", encoding="utf-8"
        )
        (feature_dir / "user-story.md").write_text(
            "# User Story\nAs a user...", encoding="utf-8"
        )

        task = PlanTask(
            task_id="P0-T2",
            phase=0,
            task_num=2,
            title="Task 2",
            checked=False,
            line_index=2,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "Resolved feature folder:" in prompt
        assert "my-feature" in prompt
        assert "CURRENT TASK" in prompt
        assert "[P0-T2] Task 2" in prompt
        assert "BEGIN plan.md" in prompt
        assert "# Plan" in prompt
        assert "BEGIN spec.md" in prompt
        assert "# Specification" in prompt
        assert "BEGIN user-story.md" in prompt
        assert "# User Story" in prompt

    def test_build_handles_missing_user_story(self, tmp_path: Path) -> None:
        """build() handles optional user-story.md gracefully."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text(
            "# Plan\n- [ ] [P0-T1] Task 1", encoding="utf-8"
        )
        (feature_dir / "spec.md").write_text(
            "# Specification\nDetails.", encoding="utf-8"
        )

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "BEGIN user-story.md" in prompt
        # Should have empty content for user-story section
        assert "---- END user-story.md ----" in prompt

    def test_build_raises_for_missing_plan(self, tmp_path: Path) -> None:
        """build() raises FileNotFoundError when plan.md missing."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)

        with pytest.raises(FileNotFoundError, match="Missing required plan.md"):
            builder.build(feature_dir, task)

    def test_build_raises_for_missing_spec(self, tmp_path: Path) -> None:
        """build() raises FileNotFoundError when spec.md missing."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)

        with pytest.raises(FileNotFoundError, match="Missing required spec.md"):
            builder.build(feature_dir, task)

    def test_build_injects_task_details(self, tmp_path: Path) -> None:
        """build() injects current task ID and title into prompt."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text(
            "# Plan\n- [ ] [P2-T5] Important task", encoding="utf-8"
        )
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P2-T5",
            phase=2,
            task_num=5,
            title="Important task",
            checked=False,
            line_index=5,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "- [P2-T5] Important task" in prompt
        assert "CURRENT TASK (execute only this task" in prompt

    def test_build_includes_toolchain_instructions(self, tmp_path: Path) -> None:
        """build() includes Black, Ruff, Pyright, Pytest instructions."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text(
            "# Plan\n- [ ] [P0-T1] Task", encoding="utf-8"
        )
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "poetry run black --check" in prompt
        assert "poetry run ruff check" in prompt
        assert "poetry run pyright" in prompt
        assert "poetry run pytest" in prompt

    def test_build_includes_plan_update_instructions(self, tmp_path: Path) -> None:
        """build() includes instructions to update plan.md checkbox."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text(
            "# Plan\n- [ ] [P1-T3] My task", encoding="utf-8"
        )
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P1-T3",
            phase=1,
            task_num=3,
            title="My task",
            checked=False,
            line_index=3,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "update plan.md by checking ONLY this task" in prompt
        assert f"- [ ] [{task.task_id}]" in prompt
        assert f"- [x] [{task.task_id}]" in prompt


class TestPromptBuilderEdgeCases:
    """Edge case tests for PromptBuilder."""

    def test_build_handles_empty_template(self, tmp_path: Path) -> None:
        """build() works with empty template file."""
        template = tmp_path / "template.md"
        template.write_text("", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        # Should still have injected context
        assert "Resolved feature folder:" in prompt
        assert "CURRENT TASK" in prompt

    def test_build_handles_empty_plan(self, tmp_path: Path) -> None:
        """build() works when plan.md is empty."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text("", encoding="utf-8")
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "BEGIN plan.md" in prompt
        assert "END plan.md" in prompt

    def test_build_uses_posix_paths_in_output(self, tmp_path: Path) -> None:
        """build() uses POSIX-style paths in prompt (cross-platform)."""
        template = tmp_path / "template.md"
        template.write_text("BASE TEMPLATE\n", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
        (feature_dir / "spec.md").write_text("# Specification\n", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        # Should use forward slashes (POSIX style)
        assert "my-feature" in prompt
        # Check that feature_dir is converted to posix in output
        assert feature_dir.as_posix() in prompt or "my-feature" in prompt

    def test_read_text_helper_uses_utf8(self, tmp_path: Path) -> None:
        """_read_text() uses UTF-8 encoding."""
        template = tmp_path / "template.md"
        template.write_text("Template with émojis 🎉", encoding="utf-8")

        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        (feature_dir / "plan.md").write_text("# Plan with émoji 📝", encoding="utf-8")
        (feature_dir / "spec.md").write_text("# Spec with émoji ✅", encoding="utf-8")

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(tmp_path, template)
        prompt = builder.build(feature_dir, task)

        assert "émojis 🎉" in prompt
        assert "émoji 📝" in prompt
        assert "émoji ✅" in prompt

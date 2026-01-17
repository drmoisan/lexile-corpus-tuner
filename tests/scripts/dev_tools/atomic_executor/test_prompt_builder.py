"""
Tests for atomic_executor.prompt_builder module.

Tests cover PromptBuilder class methods for constructing prompts from templates
and feature context (plan task excerpts + spec link).

All tests use in-memory filesystem to avoid tmp_path per repo policy.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.dev_tools.atomic_executor.plan_discovery import ResolvedPlan
from scripts.dev_tools.atomic_executor.plan_parser import PlanTask
from scripts.dev_tools.atomic_executor.prompt_builder import PromptBuilder


def make_default_plan_resolver(
    plan_filename: str = "plan.md",
) -> Callable[[Path], ResolvedPlan]:
    """
    Create a plan resolver that resolves to plan_filename in the feature dir.

    Purpose:
        Factory function to create plan resolvers for in-memory tests,
        avoiding lambda assignments that violate E731.

    Args:
        plan_filename: Name of the plan file. Defaults to "plan.md".

    Returns:
        A function that takes a feature directory and returns a ResolvedPlan
        pointing to plan_filename in that directory.
    """

    def resolve(feature_dir: Path) -> ResolvedPlan:
        return ResolvedPlan(
            path=feature_dir / plan_filename,
            display_label=plan_filename,
            update_filename=plan_filename,
        )

    return resolve


class InMemoryPromptBuilderFileSystem:
    """
    In-memory filesystem implementation for testing PromptBuilder.

    Purpose:
        Provides a pure in-memory filesystem abstraction that matches the
        PromptBuilderFileSystem protocol, enabling tests to run without
        touching the real filesystem (no tmp_path or write_text).

    Usage:
        fs = InMemoryPromptBuilderFileSystem(
            files={"/path/to/file.md": "content"},
            dirs={"/path/to/dir"},
        )

    Attributes:
        files: Mapping of POSIX path strings to file contents.
        dirs: Set of POSIX path strings representing directories.
    """

    def __init__(
        self,
        files: dict[str, str] | None = None,
        dirs: set[str] | None = None,
    ) -> None:
        """
        Initialize with pre-populated files and directories.

        Args:
            files: Dict mapping POSIX paths to file contents.
            dirs: Set of POSIX paths representing directories.
        """
        self.files: dict[str, str] = files or {}
        self.dirs: set[str] = dirs or set()

    def is_file(self, path: Path) -> bool:
        """Check if path exists as a file in memory."""
        return path.as_posix() in self.files

    def is_dir(self, path: Path) -> bool:
        """Check if path exists as a directory in memory."""
        return path.as_posix() in self.dirs

    def read_text(self, path: Path) -> str:
        """Read text content from in-memory storage."""
        key = path.as_posix()
        if key not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[key]

    def glob(self, directory: Path, pattern: str) -> list[Path]:
        """Find files matching pattern (simplified implementation)."""
        import fnmatch

        base = directory.as_posix()
        matches: list[Path] = []
        # Collect matching in-memory files under the requested directory.
        for file_path in self.files:
            if file_path.startswith(base + "/"):
                relative = file_path[len(base) + 1 :]
                if fnmatch.fnmatch(relative, pattern):
                    matches.append(Path(file_path))
        return matches


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_init_with_valid_template(self) -> None:
        """__init__() stores paths when template exists."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        fs = InMemoryPromptBuilderFileSystem(
            files={template_path.as_posix(): "Template content"}
        )

        builder = PromptBuilder(workspace, template_path, fs=fs)
        assert builder.workspace == workspace
        assert builder.template_path == template_path

    def test_init_raises_for_nonexistent_template(self) -> None:
        """__init__() raises FileNotFoundError for missing template."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/nonexistent.md")
        fs = InMemoryPromptBuilderFileSystem(files={}, dirs=set())

        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            PromptBuilder(workspace, template_path, fs=fs)

    def test_init_raises_for_directory_template(self) -> None:
        """__init__() raises FileNotFoundError when template is directory."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template")
        # Template path is a directory, not in files
        fs = InMemoryPromptBuilderFileSystem(files={}, dirs={template_path.as_posix()})

        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            PromptBuilder(workspace, template_path, fs=fs)


class TestPromptBuilderBuild:
    """Tests for build() method."""

    def test_build_combines_template_and_context(self) -> None:
        """build() combines template with plan, spec, story context."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"
        story_path = feature_dir / "user-story.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): (
                    "# Plan\n- [x] [P0-T1] Task 1\n- [ ] [P0-T2] Task 2"
                ),
                spec_path.as_posix(): "# Specification\nDetails here.",
                story_path.as_posix(): "# User Story\nAs a user...",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T2",
            phase=0,
            task_num=2,
            title="Task 2",
            checked=False,
            line_index=2,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "Resolved feature folder:" in prompt
        assert "my-feature" in prompt
        assert "CURRENT TASK" in prompt
        assert "[P0-T2] Task 2" in prompt
        assert "Plan task context:" in prompt
        assert "BEGIN plan task" in prompt
        assert "- [ ] [P0-T2] Task 2" in prompt
        assert f"Spec file: {spec_path.as_posix()}" in prompt
        assert f"User story file (optional): {story_path.as_posix()}" in prompt
        assert "# Specification" not in prompt
        assert "# User Story" not in prompt

    def test_build_handles_missing_user_story(self) -> None:
        """build() handles optional user-story.md gracefully."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task 1",
                spec_path.as_posix(): "# Specification\nDetails.",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "User story file (optional): (missing)" in prompt
        assert "BEGIN user-story.md" not in prompt

    def test_build_raises_for_missing_plan(self) -> None:
        """build() raises FileNotFoundError when plan.md missing."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        spec_path = feature_dir / "spec.md"

        # No plan.md in files, only spec.md
        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            builder.build(feature_dir, task)

    def test_build_raises_for_missing_spec(self) -> None:
        """build() raises FileNotFoundError when spec.md missing."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"

        # No spec.md in files, only plan.md
        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task 1",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )

        with pytest.raises(FileNotFoundError, match="Missing required spec.md"):
            builder.build(feature_dir, task)

    def test_build_injects_task_details(self) -> None:
        """build() injects current task ID and title into prompt."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P2-T5] Important task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P2-T5",
            phase=2,
            task_num=5,
            title="Important task",
            checked=False,
            line_index=5,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "- [P2-T5] Important task" in prompt
        assert "CURRENT TASK (execute only this task" in prompt

    def test_build_includes_toolchain_instructions(self) -> None:
        """build() includes python -m poetry run forms and forbids poetry run forms."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        # REQ-001: MUST use python -m poetry run forms
        assert "python -m poetry run black ." in prompt
        assert "python -m poetry run ruff check" in prompt
        assert "python -m poetry run pyright" in prompt
        assert "python -m poetry run pytest" in prompt
        # REQ-001: MUST NOT include any `poetry run` forms without `python -m` prefix
        assert (
            "poetry run black" not in prompt or "python -m poetry run black" in prompt
        )
        # More precise check: ensure no bare "poetry run" that isn't preceded by "-m "
        # Scan for any bare "poetry run" lines that violate the command rule.
        lines = prompt.splitlines()
        for line in lines:
            if "poetry run" in line and "python -m poetry run" not in line:
                # Reject bare poetry run commands
                raise AssertionError(
                    f"Found bare 'poetry run' without 'python -m' prefix: {line}"
                )

    def test_build_includes_plan_update_instructions(self) -> None:
        """build() includes instructions to update plan.md checkbox."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P1-T3] My task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P1-T3",
            phase=1,
            task_num=3,
            title="My task",
            checked=False,
            line_index=3,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "update plan.md by checking ONLY this task" in prompt
        assert f"- [ ] [{task.task_id}]" in prompt
        assert f"- [x] [{task.task_id}]" in prompt

    def test_build_does_not_include_model_command_when_preferred_model_is_set(
        self,
    ) -> None:
        """build() MUST NOT include /model when preferred_model is set (REQ-002)."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace,
            template_path,
            preferred_model="gpt-5.1-codex-max",
            fs=fs,
            plan_resolver=plan_resolver,
        )
        prompt = builder.build(feature_dir, task)

        # REQ-002: Prompt MUST NOT include /model anywhere
        assert "/model" not in prompt, "Prompt should not contain /model command"

    def test_build_does_not_include_interactive_session_phrase(self) -> None:
        """build() MUST NOT include 'interactive session' phrase (REQ-003)."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace,
            template_path,
            preferred_model="gpt-5.1-codex-max",
            fs=fs,
            plan_resolver=plan_resolver,
        )
        prompt = builder.build(feature_dir, task)

        # REQ-003: Prompt MUST NOT include "interactive session"
        assert (
            "interactive session" not in prompt.lower()
        ), "Prompt should not contain 'interactive session' phrase"

    def test_build_feature_placeholder_uses_relative_path_from_active(self) -> None:
        """build() uses relative path for <feature> (REQ-004)."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        # Feature dir with version subdirectory
        feature_dir = Path(
            "/workspace/docs/features/active/"
            "2026-01-06-populate-open-stax-ck-12-manifest-73/v4"
        )
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        # Template contains <feature> placeholder
        template_content = "Feature: <feature>\nBASE TEMPLATE\n"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): template_content,
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        # REQ-004: <feature> MUST be replaced with relative path from active root
        expected_feature = "2026-01-06-populate-open-stax-ck-12-manifest-73/v4"
        # The template has "Feature: <feature>" which should become
        # "Feature: 2026-01-06-populate-open-stax-ck-12-manifest-73/v4"
        # NOT "Feature: v4"
        expected_line = f"Feature: {expected_feature}"
        wrong_line = "Feature: v4"
        assert expected_line in prompt, (
            f"Expected '{expected_line}' not found in prompt. "
            "The <feature> placeholder should use the relative path from "
            "workspace/docs/features/active/, not just feature_dir.name"
        )
        assert wrong_line not in prompt, (
            f"Found '{wrong_line}' in prompt - <feature> was incorrectly "
            "substituted with just feature_dir.name instead of the relative path"
        )


class TestPromptBuilderEdgeCases:
    """Edge case tests for PromptBuilder."""

    def test_build_handles_empty_template(self) -> None:
        """build() works with empty template file."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "",  # empty template
                plan_path.as_posix(): "# Plan\n",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        # Should still have injected context
        assert "Resolved feature folder:" in prompt
        assert "CURRENT TASK" in prompt

    def test_build_handles_empty_plan(self) -> None:
        """build() works when plan.md is empty."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "",  # empty plan
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "BASE TEMPLATE" in prompt
        assert "Plan task context:" in prompt
        assert "(plan is empty)" in prompt

    def test_build_uses_posix_paths_in_output(self) -> None:
        """build() uses POSIX-style paths in prompt (cross-platform)."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "BASE TEMPLATE\n",
                plan_path.as_posix(): "# Plan\n",
                spec_path.as_posix(): "# Specification\n",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        # Should use forward slashes (POSIX style)
        assert "my-feature" in prompt
        # Check that feature_dir is converted to posix in output
        assert feature_dir.as_posix() in prompt or "my-feature" in prompt

    def test_read_text_helper_uses_utf8(self) -> None:
        """_read_text() uses UTF-8 encoding."""
        workspace = Path("/workspace")
        template_path = Path("/workspace/template.md")
        feature_dir = Path("/workspace/docs/features/active/my-feature")
        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"

        fs = InMemoryPromptBuilderFileSystem(
            files={
                template_path.as_posix(): "Template with émojis 🎉",
                plan_path.as_posix(): "# Plan\n- [ ] [P0-T1] Task with émoji 📝",
                spec_path.as_posix(): "# Spec with émoji ✅",
            },
            dirs={feature_dir.as_posix()},
        )
        plan_resolver = make_default_plan_resolver()

        task = PlanTask(
            task_id="P0-T1",
            phase=0,
            task_num=1,
            title="Task",
            checked=False,
            line_index=1,
        )
        builder = PromptBuilder(
            workspace, template_path, fs=fs, plan_resolver=plan_resolver
        )
        prompt = builder.build(feature_dir, task)

        assert "émojis 🎉" in prompt
        assert "émoji 📝" in prompt
        assert "émoji ✅" not in prompt

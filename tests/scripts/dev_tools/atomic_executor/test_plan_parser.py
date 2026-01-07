"""
Tests for atomic_executor.plan_parser module.

Tests cover PlanTask, PlanModel dataclasses, and PlanParser class methods
including parsing, task lookup, checkbox manipulation, and validation.
"""

from pathlib import Path

import pytest

from scripts.dev_tools.atomic_executor.plan_parser import (
    PlanModel,
    PlanParser,
    PlanTask,
)


class TestPlanTask:
    """Tests for PlanTask dataclass."""

    def test_plan_task_creation_with_all_fields(self) -> None:
        """PlanTask can be created with all required fields."""
        task = PlanTask(
            task_id="P1-T2",
            phase=1,
            task_num=2,
            title="Example task",
            checked=True,
            line_index=10,
        )
        assert task.task_id == "P1-T2"
        assert task.phase == 1
        assert task.task_num == 2
        assert task.title == "Example task"
        assert task.checked is True
        assert task.line_index == 10

    def test_plan_task_immutability_frozen(self) -> None:
        """PlanTask is frozen and cannot be modified after creation."""
        task = PlanTask("P1-T1", 1, 1, "Task", False, 5)
        with pytest.raises(AttributeError):
            task.checked = True  # type: ignore[misc]


class TestPlanModel:
    """Tests for PlanModel dataclass."""

    def test_plan_model_creation_empty(self) -> None:
        """PlanModel can be created with empty tasks and phases."""
        model = PlanModel(tasks=[], phases=[])
        assert model.tasks == []
        assert model.phases == []

    def test_plan_model_creation_with_data(self) -> None:
        """PlanModel can be created with tasks and phases."""
        task1 = PlanTask("P1-T1", 1, 1, "First", False, 0)
        task2 = PlanTask("P1-T2", 1, 2, "Second", True, 1)
        model = PlanModel(tasks=[task1, task2], phases=[1])
        assert len(model.tasks) == 2
        assert model.phases == [1]


class TestPlanParserInit:
    """Tests for PlanParser initialization."""

    def test_init_with_valid_file(self, tmp_path: Path) -> None:
        """PlanParser initializes successfully with valid file path."""
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Plan\n", encoding="utf-8")

        parser = PlanParser(plan_file)
        assert parser.plan_path == plan_file

    def test_init_with_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """PlanParser raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "missing.md"
        with pytest.raises(FileNotFoundError, match="Plan file not found"):
            PlanParser(nonexistent)


class TestPlanParserParse:
    """Tests for PlanParser.parse() method."""

    def test_parse_empty_file_returns_empty_model(self, tmp_path: Path) -> None:
        """Parsing an empty file returns PlanModel with no tasks or phases."""
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("", encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        assert model.tasks == []
        assert model.phases == []

    def test_parse_single_unchecked_task(self, tmp_path: Path) -> None:
        """Parsing a single unchecked task returns correct PlanTask."""
        plan_file = tmp_path / "plan.md"
        plan_content = "## Phase 1\n- [ ] [P1-T1] First task\n"
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        assert len(model.tasks) == 1
        assert model.phases == [1]

        task = model.tasks[0]
        assert task.task_id == "P1-T1"
        assert task.phase == 1
        assert task.task_num == 1
        assert task.title == "First task"
        assert task.checked is False
        assert task.line_index == 1

    def test_parse_single_checked_task(self, tmp_path: Path) -> None:
        """Parsing a single checked task returns correct PlanTask."""
        plan_file = tmp_path / "plan.md"
        plan_content = "## Phase 1\n- [x] [P1-T1] First task\n"
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        assert len(model.tasks) == 1

        task = model.tasks[0]
        assert task.checked is True

    def test_parse_multiple_tasks_multiple_phases(self, tmp_path: Path) -> None:
        """Parsing multiple tasks across phases returns all correctly."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 0: Setup
- [ ] [P0-T1] Setup task

## Phase 1: Implementation
- [x] [P1-T1] First impl
- [ ] [P1-T2] Second impl

## Phase 2: Final checks
- [ ] [P2-T1] QA task
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        assert len(model.tasks) == 4
        assert model.phases == [0, 1, 2]

        # Validate P0-T1
        assert model.tasks[0].task_id == "P0-T1"
        assert model.tasks[0].phase == 0
        assert model.tasks[0].checked is False

        # Validate P1-T1
        assert model.tasks[1].task_id == "P1-T1"
        assert model.tasks[1].phase == 1
        assert model.tasks[1].checked is True

        # Validate P1-T2
        assert model.tasks[2].task_id == "P1-T2"
        assert model.tasks[2].phase == 1
        assert model.tasks[2].checked is False

        # Validate P2-T1
        assert model.tasks[3].task_id == "P2-T1"
        assert model.tasks[3].phase == 2
        assert model.tasks[3].checked is False

    def test_parse_ignores_non_task_lines(self, tmp_path: Path) -> None:
        """Parser ignores lines that don't match task pattern."""
        plan_file = tmp_path / "plan.md"
        plan_content = """# Plan Document

Some intro text.

## Phase 1
- [ ] [P1-T1] Valid task

Random text without checkbox.
- Not a task (no checkbox)
- [ ] Missing task ID

## Phase 2
- [x] [P2-T1] Another valid
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        # Only P1-T1 and P2-T1 should be captured
        assert len(model.tasks) == 2
        assert model.tasks[0].task_id == "P1-T1"
        assert model.tasks[1].task_id == "P2-T1"


class TestPlanParserNextUncheckedTask:
    """Tests for PlanParser.next_unchecked_task() method."""

    def test_next_unchecked_task_returns_first_unchecked(self, tmp_path: Path) -> None:
        """next_unchecked_task returns the first unchecked task."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Done
- [ ] [P1-T2] Not done
- [ ] [P1-T3] Also not done
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        task = parser.next_unchecked_task()
        assert task is not None
        assert task.task_id == "P1-T2"

    def test_next_unchecked_task_returns_none_when_all_checked(
        self, tmp_path: Path
    ) -> None:
        """next_unchecked_task returns None when all tasks are checked."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Done
- [x] [P1-T2] Also done
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        task = parser.next_unchecked_task()
        assert task is None

    def test_next_unchecked_task_returns_none_for_empty_model(
        self, tmp_path: Path
    ) -> None:
        """next_unchecked_task returns None for empty task list."""
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("", encoding="utf-8")

        parser = PlanParser(plan_file)
        task = parser.next_unchecked_task()
        assert task is None


class TestPlanParserFindTaskById:
    """Tests for PlanParser.find_task_by_id() method."""

    def test_find_task_by_id_returns_matching_task(self, tmp_path: Path) -> None:
        """find_task_by_id returns the task with matching ID."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [ ] [P1-T1] First
- [ ] [P1-T2] Second
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        task = parser.find_task_by_id("P1-T2")
        assert task is not None
        assert task.task_id == "P1-T2"
        assert task.title == "Second"

    def test_find_task_by_id_raises_for_missing_id(self, tmp_path: Path) -> None:
        """find_task_by_id raises RuntimeError when ID not found."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [ ] [P1-T1] First
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        with pytest.raises(RuntimeError, match="Task id not found"):
            parser.find_task_by_id("P1-T99")


class TestPlanParserPhaseComplete:
    """Tests for PlanParser.phase_complete() method."""

    def test_phase_complete_returns_true_when_all_checked(self, tmp_path: Path) -> None:
        """phase_complete returns True when all tasks in phase are checked."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Done
- [x] [P1-T2] Also done
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        assert parser.phase_complete(1) is True

    def test_phase_complete_returns_false_when_any_unchecked(
        self, tmp_path: Path
    ) -> None:
        """phase_complete returns False when any task in phase is unchecked."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Done
- [ ] [P1-T2] Not done
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        assert parser.phase_complete(1) is False

    def test_phase_complete_returns_false_for_nonexistent_phase(
        self, tmp_path: Path
    ) -> None:
        """phase_complete returns False for phase with no tasks."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Done
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        # Phase 99 doesn't exist, so returns False
        assert parser.phase_complete(99) is False


class TestPlanParserFlipCheckbox:
    """Tests for PlanParser.flip_checkbox() method."""

    def test_flip_checkbox_checks_unchecked_task(self, tmp_path: Path) -> None:
        """flip_checkbox changes [ ] to [x] for the specified task."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [ ] [P1-T1] First task
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        task = model.tasks[0]
        parser.flip_checkbox(task)

        # Read back and verify
        updated_content = plan_file.read_text(encoding="utf-8")
        assert "- [x] [P1-T1] First task" in updated_content

    def test_flip_checkbox_is_idempotent_for_checked_task(self, tmp_path: Path) -> None:
        """flip_checkbox is idempotent for already-checked tasks."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] First task
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        task = model.tasks[0]
        parser.flip_checkbox(task)

        # Read back and verify - should still be checked
        updated_content = plan_file.read_text(encoding="utf-8")
        assert "- [x] [P1-T1] First task" in updated_content

    def test_flip_checkbox_preserves_other_lines(self, tmp_path: Path) -> None:
        """flip_checkbox only modifies the target line, leaving others intact."""
        plan_file = tmp_path / "plan.md"
        plan_content = """# Plan Document

## Phase 1
- [ ] [P1-T1] First task
- [ ] [P1-T2] Second task

Some footer text.
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        task = model.tasks[0]  # P1-T1
        parser.flip_checkbox(task)

        updated_content = plan_file.read_text(encoding="utf-8")
        # P1-T1 should be checked
        assert "- [x] [P1-T1] First task" in updated_content
        # P1-T2 should remain unchecked
        assert "- [ ] [P1-T2] Second task" in updated_content
        # Other content preserved
        assert "# Plan Document" in updated_content
        assert "Some footer text." in updated_content


class TestPlanParserPreflightValidate:
    """Tests for PlanParser.preflight_validate() method."""

    def test_preflight_validate_passes_with_phase_0_and_qa(
        self, tmp_path: Path
    ) -> None:
        """preflight_validate succeeds when Phase 0 and QA/toolchain present."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 0: Setup
- [ ] [P0-T1] Setup

## Phase 1: Implementation
- [ ] [P1-T1] Impl

## Phase 2: QA
- [ ] [P2-T1] Run Black, Ruff, Pyright, Pytest
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        # Should not raise
        parser.preflight_validate()

    def test_preflight_validate_raises_when_phase_0_missing(
        self, tmp_path: Path
    ) -> None:
        """preflight_validate raises ValueError when Phase 0 is missing."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1: Implementation
- [ ] [P1-T1] Impl

## Phase QA: Final
- [ ] [PQA-T1] QA
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        with pytest.raises(RuntimeError, match="missing Phase 0"):
            parser.preflight_validate()

    def test_preflight_validate_raises_when_qa_phase_missing(
        self, tmp_path: Path
    ) -> None:
        """preflight_validate raises ValueError when QA phase is missing."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 0: Setup
- [ ] [P0-T1] Setup

## Phase 1: Implementation
- [ ] [P1-T1] Impl
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        with pytest.raises(RuntimeError, match="QA/toolchain"):
            parser.preflight_validate()

    def test_preflight_validate_raises_when_both_missing(self, tmp_path: Path) -> None:
        """preflight_validate raises RuntimeError when both Phase 0 and QA missing."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1: Implementation
- [ ] [P1-T1] Impl
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        # Should raise for Phase 0 first (checked in order)
        with pytest.raises(RuntimeError, match="missing Phase 0"):
            parser.preflight_validate()


class TestPlanParserEdgeCases:
    """Edge case tests for PlanParser."""

    def test_parse_handles_mixed_checkbox_formats(self, tmp_path: Path) -> None:
        """Parser correctly handles [x], [X], [ ] variants."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [x] [P1-T1] Lowercase x
- [X] [P1-T2] Uppercase X
- [ ] [P1-T3] Unchecked
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        assert len(model.tasks) == 3
        assert model.tasks[0].checked is True
        assert model.tasks[1].checked is True
        assert model.tasks[2].checked is False

    def test_parse_handles_whitespace_variations(self, tmp_path: Path) -> None:
        """Parser handles variations in whitespace around task elements."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
-  [x]  [P1-T1]  Task with extra spaces
- [ ] [P1-T2]No space after ID
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        # Should still parse both tasks
        assert len(model.tasks) == 2

    def test_flip_checkbox_is_idempotent(self, tmp_path: Path) -> None:
        """Flipping checkbox multiple times is idempotent - stays checked."""
        plan_file = tmp_path / "plan.md"
        plan_content = """## Phase 1
- [ ] [P1-T1] Task
"""
        plan_file.write_text(plan_content, encoding="utf-8")

        parser = PlanParser(plan_file)
        model = parser.parse()
        task = model.tasks[0]

        # Flip twice
        parser.flip_checkbox(task)
        model_after_first = parser.parse()
        task_after_first = model_after_first.tasks[0]
        parser.flip_checkbox(task_after_first)

        # Should still be checked (idempotent)
        final_model = parser.parse()
        assert final_model.tasks[0].checked is True

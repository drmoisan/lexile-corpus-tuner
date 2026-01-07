"""
Tests for atomic_executor.cli module (new implementation).

Covers single-task execution with retries, execute-all orchestration,
and logging/prompt enhancements.
"""

# pyright: reportArgumentType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from scripts.dev_tools.atomic_executor.cli import main
from scripts.dev_tools.atomic_executor.plan_parser import PlanTask


@pytest.fixture
def mock_dependencies() -> Generator[dict[str, Any], None, None]:
    """Mock external dependencies for CLI integration tests."""
    with (
        patch("scripts.dev_tools.atomic_executor.cli.PlanParser") as MockParser,
        patch("scripts.dev_tools.atomic_executor.cli.FeatureResolver") as MockResolver,
        patch("scripts.dev_tools.atomic_executor.cli.QCRunner") as MockQCRunner,
        patch("scripts.dev_tools.atomic_executor.cli.PromptBuilder") as MockBuilder,
        patch("scripts.dev_tools.atomic_executor.cli.copy_to_clipboard") as MockClip,
        patch("scripts.dev_tools.atomic_executor.cli.run_copilot") as MockRunCopilot,
        patch("scripts.dev_tools.atomic_executor.cli.ensure_clean_tree"),
        patch("scripts.dev_tools.atomic_executor.cli.refuse_protected_branch"),
        patch("pathlib.Path.is_file") as MockIsFile,
    ):

        # Setup common mock behaviors
        MockIsFile.return_value = True

        parser_instance = MockParser.return_value
        # Setup plan with one unchecked task
        task = PlanTask(
            task_id="P1-T1",
            phase=1,
            task_num=1,
            title="Test task",
            checked=False,
            line_index=10,
        )
        parser_instance.next_unchecked_task.return_value = task
        parser_instance.find_task_by_id.return_value = task
        parser_instance.phase_complete.return_value = False

        resolver_instance = MockResolver.return_value
        resolver_instance.resolve.return_value = (Mock(), Path("/mock/feature/dir"))

        yield {
            "parser_cls": MockParser,
            "parser": parser_instance,
            "resolver": resolver_instance,
            "qc_runner": MockQCRunner,
            "builder": MockBuilder,
            "clip": MockClip,
            "run_copilot": MockRunCopilot,
            "task": task,
        }


def test_execute_one_task_retries_until_success(mock_dependencies: dict[str, Any]):
    """
    [P1-T1] Test that execute command retries on QC failure and succeeds eventually.

    Verifies:
    1. QC runs and fails initially.
    2. Copilot run repeats.
    3. QC runs and succeeds.
    4. Checkbox is flipped.
    """
    mocks = mock_dependencies

    # Setup QC to fail once then succeed
    qc_instance = mocks["qc_runner"].return_value
    qc_instance.run_scoped.side_effect = [
        subprocess.CalledProcessError(1, "qc"),  # Attempt 1 fail
        None,  # Attempt 2 succeed
    ]

    argv = ["execute", "feature-folder", "--max-fix-attempts", "3"]
    exit_code = main(argv)

    assert exit_code == 0

    # Copilot should run twice (1st attempt + 1 retry)
    assert mocks["run_copilot"].call_count == 2

    # QC should run twice
    assert qc_instance.run_scoped.call_count == 2

    # Checkbox should be folded on success
    mocks["parser"].flip_checkbox.assert_called_once_with(mocks["task"])


def test_execute_all_runs_full_qc_after_phase(mock_dependencies: dict[str, Any]):
    """
    [P2-T2] Test that execute-all runs full QC when a phase is complete.

    Setup:
    - 2 tasks in plan: P1-T1 (unchecked), P1-T2 (unchecked)
    - P1-T1 completes -> parser says phase 1 NOT complete
    - P1-T2 completes -> parser says phase 1 COMPLETE

    Verifies:
    - full_qc runs after P1-T2
    """
    mocks = mock_dependencies
    parser = mocks["parser"]

    # 2 tasks
    task1 = PlanTask(
        phase=1, task_num=1, title="T1", checked=False, line_index=1, task_id="P1-T1"
    )
    task2 = PlanTask(
        phase=1, task_num=2, title="T2", checked=False, line_index=2, task_id="P1-T2"
    )

    # next_unchecked_task sequence: T1, T2, None
    parser.next_unchecked_task.side_effect = [task1, task2, None]
    parser.find_task_by_id.side_effect = [task1, task2]  # for find by ID

    # phase_complete sequence: T1->False, T2->True
    parser.phase_complete.side_effect = [False, True]

    argv = ["execute-all", "feature-folder"]
    exit_code = main(argv)

    assert exit_code == 0
    # Copilot called twice
    assert mocks["run_copilot"].call_count == 2
    # Full QC called once (after T2)
    mocks["qc_runner"].return_value.run_full.assert_called_once()


def test_execute_all_respects_infinite_retry(mock_dependencies: dict[str, Any]):
    """
    [P2-T3] Test that max-fix-attempts=0 means infinite retry.

    Setup:
    - 1 tasks in plan: P1-T1 (unchecked)
    - QC fails 3 times, succeeds on 4th.
    - max_fix_attempts = 0 (infinite)

    Verifies:
    - Copilot runs 4 times.
    - Exit code 0.
    """
    mocks = mock_dependencies

    # Configure parser to return task on first call, then None (done)
    task = mocks["task"]
    mocks["parser"].next_unchecked_task.side_effect = [task, None]

    # Setup QC to fail 3 times then succeed
    qc_instance = mocks["qc_runner"].return_value
    qc_instance.run_scoped.side_effect = [
        subprocess.CalledProcessError(1, "qc"),  # Attempt 1 fail
        subprocess.CalledProcessError(1, "qc"),  # Attempt 2 fail
        subprocess.CalledProcessError(1, "qc"),  # Attempt 3 fail
        None,  # Attempt 4 succeed
    ]

    argv = ["execute-all", "feature-folder", "--max-fix-attempts", "0"]
    exit_code = main(argv)

    assert exit_code == 0
    assert mocks["run_copilot"].call_count == 4
    # Checkbox should be folded
    mocks["parser"].flip_checkbox.assert_called_once()


def test_execute_all_aborts_with_exit_code_5_on_persistent_failure(
    mock_dependencies: dict[str, Any],
):
    """
    [P3-T3] Verify execute-all aborts if a task persistently fails (returns 5).
    """
    mocks = mock_dependencies

    # Define tasks
    task1 = PlanTask(
        phase=1,
        task_num=1,
        task_id="P1-T1",
        title="Task 1",
        checked=False,
        line_index=10,
    )
    task2 = PlanTask(
        phase=1,
        task_num=2,
        task_id="P1-T2",
        title="Task 2",
        checked=False,
        line_index=11,
    )

    # Configure parser to return Task1, then Task2, then potentially Task3 (if called)
    # 1. Initial next_unchecked_task() call -> Task1
    # 2. Loop end next_unchecked_task() call -> Task2
    # 3. Task2 fails -> Loop exits, so no more calls
    mocks["parser"].next_unchecked_task.side_effect = [task1, task2]
    mocks["parser"].phase_complete.return_value = False

    # Mock _execute_one_task to return 0 (Task1 success) then 5 (Task2 recurring fail)
    with patch("scripts.dev_tools.atomic_executor.cli._execute_one_task") as mock_exec:
        mock_exec.side_effect = [0, 5]

        # In execute-all mode
        argv = ["execute-all", "feat"]

        # ACT
        exit_code = main(argv)

        # ASSERT
        assert exit_code == 5
        assert mock_exec.call_count == 2

        # Verify calls match tasks
        _, kwargs1 = mock_exec.call_args_list[0]
        assert kwargs1["cur"] == task1

        _, kwargs2 = mock_exec.call_args_list[1]
        assert kwargs2["cur"] == task2

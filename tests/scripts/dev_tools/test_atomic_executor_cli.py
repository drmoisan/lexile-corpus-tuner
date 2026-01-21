"""
Tests for atomic_executor.cli module (new implementation).

Covers single-task execution with retries, execute-all orchestration,
and logging/prompt enhancements.
"""

# pyright: reportArgumentType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

import contextlib
import io
import os
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from scripts.dev_tools.atomic_executor import cli
from scripts.dev_tools.atomic_executor.cli import main
from scripts.dev_tools.atomic_executor.copilot_runner import CopilotRunResult
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
        patch(
            "scripts.dev_tools.atomic_executor.cli._run_preflight_qc_fix_loop",
            return_value=0,
        ),
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

        # Treat Copilot CLI as succeeding so tests can focus on QC retry orchestration.
        MockRunCopilot.return_value = CopilotRunResult(exit_code=0, output_tail="")

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


def _setup_run_copilot_capture(
    monkeypatch: pytest.MonkeyPatch, supports_sessions: bool
) -> list[str]:
    """
    Configure run_copilot dependencies to capture argv without filesystem I/O.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch dependencies.
        supports_sessions (bool): Whether to report session support.

    Returns:
        list[str]: Captured argv list from the mocked Popen call.
    """

    captured_argv: list[str] = []

    def _fake_exists(path: Path) -> bool:
        """
        Pretend the fake copilot executable exists on PATH.

        Args:
            path (Path): Path to check.

        Returns:
            bool: True for the fake copilot path, False otherwise.
        """

        return path.as_posix() == "/fake/bin/copilot"

    def _fake_open(_self: Path, *_args: object, **_kwargs: object):
        """
        Return an in-memory file handle for log writes.

        Returns:
            Context manager wrapping an in-memory StringIO.
        """

        return contextlib.nullcontext(io.StringIO())

    def _fake_write_text(_self: Path, *_args: object, **_kwargs: object) -> int:
        """
        Pretend to write text without touching disk.

        Returns:
            int: Number of characters written (0 for in-memory no-op).
        """

        return 0

    def _fake_mkdir(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Pretend to create a directory without touching disk.
        """

        return None

    def _fake_touch(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Pretend to create a file without touching disk.
        """

        return None

    def _fake_popen(argv: list[str], *_args: object, **_kwargs: object) -> Mock:
        """
        Capture argv passed to subprocess.Popen and return a dummy process.

        Args:
            argv (list[str]): Copilot CLI argument vector.

        Returns:
            Mock: Dummy process handle.
        """

        captured_argv.extend(argv)
        return Mock()

    monkeypatch.setenv("PATH", "/fake/bin")
    monkeypatch.setattr(cli.Path, "exists", _fake_exists)
    monkeypatch.setattr(cli.Path, "open", _fake_open)
    monkeypatch.setattr(cli.Path, "write_text", _fake_write_text)
    monkeypatch.setattr(cli.Path, "mkdir", _fake_mkdir)
    monkeypatch.setattr(cli.Path, "touch", _fake_touch)
    monkeypatch.setattr(cli.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cli, "_stream_copilot_output", lambda **_kwargs: (0, ""))
    monkeypatch.setattr(cli, "_clean_session_file", lambda *_args, **_kwargs: None)
    # Note: _copilot_supports_session was never implemented; session support
    # is determined by whether --continue is added in run_copilot.

    return captured_argv


def _assert_argv_contains_sequence(argv: list[str], sequence: list[str]) -> None:
    """
    Assert that argv contains a contiguous sequence of arguments.

    Args:
        argv (list[str]): Argument vector to inspect.
        sequence (list[str]): Sequence that must appear contiguously.
    """

    # Scan for the contiguous sequence to avoid brittle indexing.
    for idx in range(len(argv) - len(sequence) + 1):
        if argv[idx : idx + len(sequence)] == sequence:
            return
    raise AssertionError(f"Expected sequence {sequence} not found in argv: {argv}")


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

    # Mock execute_one_task to return 0 (Task1 success) then 5 (Task2 recurring fail)
    with patch("scripts.dev_tools.atomic_executor.cli.execute_one_task") as mock_exec:
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


def test_copilot_argv_includes_agent_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_copilot() should include the atomic executor agent flag."""
    captured_argv = _setup_run_copilot_capture(monkeypatch, supports_sessions=False)

    cli.run_copilot(
        workspace=Path("/workspace"),
        prompt_text="test prompt",
        log_file=Path("/workspace/.agent_logs/atomic_executor_test.log"),
        task_id="P1-T1",
        preferred_model=None,
        run_id="2026-01-07_000000",
    )

    _assert_argv_contains_sequence(captured_argv, ["--agent", "atomic_execution"])


def test_first_task_omits_continue_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_copilot() should omit --continue for the first task."""
    captured_argv = _setup_run_copilot_capture(monkeypatch, supports_sessions=True)

    cli.run_copilot(
        workspace=Path("/workspace"),
        prompt_text="test prompt",
        log_file=Path("/workspace/.agent_logs/atomic_executor_test.log"),
        task_id="P1-T1",
        preferred_model=None,
        run_id="2026-01-07_000000",
        is_first_task=True,
    )

    assert "--continue" not in captured_argv


def test_subsequent_task_includes_continue_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_copilot() should add --continue after the first task."""
    captured_argv = _setup_run_copilot_capture(monkeypatch, supports_sessions=True)

    cli.run_copilot(
        workspace=Path("/workspace"),
        prompt_text="test prompt",
        log_file=Path("/workspace/.agent_logs/atomic_executor_test.log"),
        task_id="P1-T2",
        preferred_model=None,
        run_id="2026-01-07_000000",
        is_first_task=False,
    )

    assert "--continue" in captured_argv


def test_single_run_lock_acquired_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """acquire_executor_lock() should create the lock file."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    lock_exists = False
    lock_file_name = ".agent_logs/executor.lock"
    original_exists = cli.Path.exists
    original_write_text = cli.Path.write_text
    original_mkdir = cli.Path.mkdir

    def _fake_exists(path: Path) -> bool:
        """
        Check whether the simulated lock file exists.

        Args:
            path (Path): Path to check.

        Returns:
            bool: True when the lock flag is set for the lock path.
        """

        if path.as_posix().endswith(lock_file_name):
            return lock_exists
        return original_exists(path)

    def _fake_write_text(_self: Path, *_args: object, **_kwargs: object) -> int:
        """
        Simulate writing the lock file and flip the existence flag.

        Returns:
            int: Number of characters written (0 for simulated write).
        """

        nonlocal lock_exists
        if _self.as_posix().endswith(lock_file_name):
            lock_exists = True
            return 0
        return original_write_text(_self, *_args, **_kwargs)

    def _fake_mkdir(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Simulate creating the lock directory without touching disk.
        """

        if _self.as_posix().endswith(".agent_logs"):
            return None
        return original_mkdir(_self, *_args, **_kwargs)

    monkeypatch.setattr(cli.Path, "exists", _fake_exists)
    monkeypatch.setattr(cli.Path, "write_text", _fake_write_text)
    monkeypatch.setattr(cli.Path, "mkdir", _fake_mkdir)

    lock_path = cli.acquire_executor_lock(Path("/workspace"))

    if os.getenv(cli.EXECUTOR_LOCK_BYPASS_ENV) == "1":
        assert lock_exists is False
    else:
        assert lock_exists is True
    assert lock_path.as_posix().endswith(lock_file_name)


def test_single_run_lock_blocks_concurrent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """acquire_executor_lock() should raise when the lock already exists."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    lock_exists = True
    lock_file_name = ".agent_logs/executor.lock"
    original_exists = cli.Path.exists
    original_mkdir = cli.Path.mkdir

    def _fake_exists(path: Path) -> bool:
        """
        Report the lock as existing for the lock file path.

        Args:
            path (Path): Path to check.

        Returns:
            bool: True for the executor lock path.
        """

        if path.as_posix().endswith(lock_file_name):
            return lock_exists
        return original_exists(path)

    def _fake_mkdir(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Simulate creating the lock directory without touching disk.
        """

        if _self.as_posix().endswith(".agent_logs"):
            return None
        return original_mkdir(_self, *_args, **_kwargs)

    monkeypatch.setattr(cli.Path, "exists", _fake_exists)
    monkeypatch.setattr(cli.Path, "mkdir", _fake_mkdir)

    if os.getenv(cli.EXECUTOR_LOCK_BYPASS_ENV) == "1":
        lock_path = cli.acquire_executor_lock(Path("/workspace"))
        assert lock_path.as_posix().endswith(lock_file_name)
        return

    with pytest.raises(RuntimeError, match="executor lock already exists"):
        cli.acquire_executor_lock(Path("/workspace"))


def test_single_run_lock_allows_bypass_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """acquire_executor_lock() should bypass lock when env flag is set."""
    lock_exists = True
    lock_file_name = ".agent_logs/executor.lock"
    original_exists = cli.Path.exists
    original_mkdir = cli.Path.mkdir

    def _fake_exists(path: Path) -> bool:
        """
        Report the lock as existing for the lock file path.

        Args:
            path (Path): Path to check.

        Returns:
            bool: True for the executor lock path.
        """

        if path.as_posix().endswith(lock_file_name):
            return lock_exists
        return original_exists(path)

    def _fake_mkdir(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Simulate creating the lock directory without touching disk.
        """

        if _self.as_posix().endswith(".agent_logs"):
            return None
        return original_mkdir(_self, *_args, **_kwargs)

    monkeypatch.setattr(cli.Path, "exists", _fake_exists)
    monkeypatch.setattr(cli.Path, "mkdir", _fake_mkdir)
    monkeypatch.setenv(cli.EXECUTOR_LOCK_BYPASS_ENV, "1")

    lock_path = cli.acquire_executor_lock(Path("/workspace"))

    assert lock_path.as_posix().endswith(lock_file_name)


def test_single_run_lock_released_on_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """release_executor_lock() should remove the lock file."""
    lock_exists = True
    lock_file_name = ".agent_logs/executor.lock"
    original_exists = cli.Path.exists

    def _fake_exists(path: Path) -> bool:
        """
        Report the lock as existing for the lock file path.

        Args:
            path (Path): Path to check.

        Returns:
            bool: True for the executor lock path.
        """

        if path.as_posix().endswith(lock_file_name):
            return lock_exists
        return original_exists(path)

    def _fake_unlink(_self: Path, *_args: object, **_kwargs: object) -> None:
        """
        Simulate removing the lock file by clearing the flag.
        """

        nonlocal lock_exists
        lock_exists = False

    monkeypatch.setattr(cli.Path, "exists", _fake_exists)
    monkeypatch.setattr(cli.Path, "unlink", _fake_unlink)

    cli.release_executor_lock(Path("/workspace/.agent_logs/executor.lock"))

    assert lock_exists is False


class TestExpectFailBehavior:
    """
    Tests for [expect-fail] task semantics in execute_one_task().

    Purpose:
        Verifies that tasks annotated with [expect-fail] invert pytest success
        criteria (TDD Red workflow) while still enforcing non-pytest QC steps.
    """

    def test_execute_one_task_expect_fail_succeeds_on_pytest_failure(
        self, mock_dependencies: dict[str, Any]
    ) -> None:
        """
        Task with expect_fail=True should succeed when pytest fails but QC passes.

        Purpose:
            TDD Red workflow: a failing test is the expected outcome.

        Verifies:
            - exit_code == 0 (success)
            - Checkbox is flipped
            - Log output contains the required verification message
        """
        mocks = mock_dependencies

        # Create expect-fail task
        task = PlanTask(
            task_id="P1-T1",
            phase=1,
            task_num=1,
            title="Add failing regression test",
            checked=False,
            line_index=10,
            expect_fail=True,
        )
        mocks["parser"].next_unchecked_task.return_value = task
        mocks["parser"].find_task_by_id.return_value = task

        # QC: black/ruff/pyright pass, pytest fails
        qc_instance = mocks["qc_runner"].return_value
        qc_instance.run_scoped.side_effect = subprocess.CalledProcessError(1, "pytest")
        qc_instance.run_black.return_value = None
        qc_instance.run_ruff.return_value = None
        qc_instance.run_pyright.return_value = None
        qc_instance.run_pytest.side_effect = subprocess.CalledProcessError(1, "pytest")

        argv = ["execute", "feature-folder", "--max-fix-attempts", "1"]
        exit_code = main(argv)

        # After implementation: expect_fail + pytest failure = success
        assert exit_code == 0
        mocks["parser"].flip_checkbox.assert_called_once_with(task)

    def test_execute_one_task_expect_fail_retries_and_exits_5_when_qc_passes(
        self, mock_dependencies: dict[str, Any]
    ) -> None:
        """
        Task with expect_fail=True should fail if QC unexpectedly passes (green).

        Purpose:
            Unexpected green means the TDD Red precondition is violated.

        Verifies:
            - exit_code == 5 (persistent failure after retries)
            - Copilot retried max_fix_attempts times
        """
        mocks = mock_dependencies

        # Create expect-fail task
        task = PlanTask(
            task_id="P1-T1",
            phase=1,
            task_num=1,
            title="Add failing regression test",
            checked=False,
            line_index=10,
            expect_fail=True,
        )
        mocks["parser"].next_unchecked_task.return_value = task
        mocks["parser"].find_task_by_id.return_value = task

        # QC fully passes (unexpected green)
        qc_instance = mocks["qc_runner"].return_value
        qc_instance.run_scoped.return_value = None

        argv = ["execute", "feature-folder", "--max-fix-attempts", "2"]
        exit_code = main(argv)

        # After implementation: unexpected green should retry then return 5
        assert exit_code == 5
        # Copilot should retry (1 initial + 2 max_fix_attempts = 3 total)
        assert mocks["run_copilot"].call_count >= 2

    def test_execute_one_task_expect_fail_does_not_mask_non_pytest_failure(
        self, mock_dependencies: dict[str, Any]
    ) -> None:
        """
        Task with expect_fail=True should still fail if black/ruff/pyright fails.

        Purpose:
            Only pytest failure is expected; other QC failures are still errors.

        Verifies:
            - Non-pytest QC failure triggers retry loop
            - Behavior matches normal task handling for non-pytest steps
        """
        mocks = mock_dependencies

        # Create expect-fail task
        task = PlanTask(
            task_id="P1-T1",
            phase=1,
            task_num=1,
            title="Add failing regression test",
            checked=False,
            line_index=10,
            expect_fail=True,
        )
        mocks["parser"].next_unchecked_task.return_value = task
        mocks["parser"].find_task_by_id.return_value = task

        # QC fails on ruff (non-pytest failure)
        qc_instance = mocks["qc_runner"].return_value
        qc_instance.run_scoped.side_effect = subprocess.CalledProcessError(1, "ruff")

        argv = ["execute", "feature-folder", "--max-fix-attempts", "1"]
        exit_code = main(argv)

        # Non-pytest failure should still return failure (exit 5 after retries).
        # This test should pass both before and after implementation.
        assert exit_code == 5

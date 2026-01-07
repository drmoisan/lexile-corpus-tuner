"""
Tests for atomic_executor.qc_runner module.

Tests cover QCRunner class methods for running scoped and full QC toolchains
(Black, Ruff, Pyright, Pytest) on changed files or entire codebase.
"""

# pyright: reportPrivateUsage=false

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from scripts.dev_tools.atomic_executor.qc_runner import QCRunner

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


class TestQCRunnerInit:
    """Tests for QCRunner initialization."""

    def test_init_stores_workspace(self, tmp_path: Path) -> None:
        """__init__() stores workspace path."""
        runner = QCRunner(tmp_path)
        assert runner.workspace == tmp_path


class TestQCRunnerChangedFiles:
    """Tests for changed_files() method."""

    def test_changed_files_parses_git_status(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """changed_files() parses git status --porcelain output."""
        git_output = " M src/module.py\nA  tests/test_new.py\n D  old.py\n"

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = git_output
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        files = runner.changed_files()
        assert files == ["src/module.py", "tests/test_new.py", "old.py"]

    def test_changed_files_handles_empty_output(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """changed_files() returns empty list when no changes."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        files = runner.changed_files()
        assert files == []

    def test_changed_files_handles_malformed_lines(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """changed_files() skips malformed git status lines."""
        git_output = " M src/module.py\nmalformed_line\n?? new.py\n"

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = git_output
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        files = runner.changed_files()
        # Should include src/module.py and new.py, skip malformed_line
        assert "src/module.py" in files
        assert "new.py" in files
        assert len(files) == 2


class TestQCRunnerFilterHelpers:
    """Tests for _filter_* helper methods."""

    def test_filter_python_files_keeps_py_only(self, tmp_path: Path) -> None:
        """_filter_python_files() keeps only .py files."""
        runner = QCRunner(tmp_path)
        files = ["src/module.py", "tests/test.py", "README.md", "config.yaml"]
        result = runner._filter_python_files(
            files
        )  # pyright: ignore[reportPrivateUsage]
        assert result == ["src/module.py", "tests/test.py"]

    def test_filter_python_files_returns_empty_for_no_py(self, tmp_path: Path) -> None:
        """_filter_python_files() returns empty list when no .py files."""
        runner = QCRunner(tmp_path)
        files = ["README.md", "config.yaml", "data.json"]
        result = runner._filter_python_files(
            files
        )  # pyright: ignore[reportPrivateUsage]
        assert result == []

    def test_filter_test_files_keeps_tests_only(self, tmp_path: Path) -> None:
        """_filter_test_files() keeps only files in tests/ directories."""
        runner = QCRunner(tmp_path)
        files = [
            "tests/test_module.py",
            "src/tests/test_helper.py",
            "src/module.py",
            "test_outside.py",
        ]
        result = runner._filter_test_files(files)  # pyright: ignore[reportPrivateUsage]
        assert result == ["tests/test_module.py", "src/tests/test_helper.py"]

    def test_filter_test_files_requires_py_extension(self, tmp_path: Path) -> None:
        """_filter_test_files() requires .py extension."""
        runner = QCRunner(tmp_path)
        files = ["tests/test_module.py", "tests/README.md", "tests/data.json"]
        result = runner._filter_test_files(files)  # pyright: ignore[reportPrivateUsage]
        assert result == ["tests/test_module.py"]


class TestQCRunnerRunScoped:
    """Tests for run_scoped() method."""

    def test_run_scoped_runs_all_tools_on_changed_files(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_scoped() runs Black, Ruff, Pyright, Pytest on changed files."""
        calls: list[list[str]] = []

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            result = Mock()
            result.returncode = 0
            result.stdout = " M src/module.py\n M tests/test_module.py\n"
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        runner.run_scoped()

        # Should have called git status, black, ruff, pyright, pytest
        assert len(calls) == 5
        assert calls[0] == ["git", "status", "--porcelain"]
        assert calls[1] == [
            "poetry",
            "run",
            "black",
            "--check",
            "src/module.py",
            "tests/test_module.py",
        ]
        assert calls[2] == [
            "poetry",
            "run",
            "ruff",
            "check",
            "src/module.py",
            "tests/test_module.py",
        ]
        assert calls[3] == [
            "poetry",
            "run",
            "pyright",
            "src/module.py",
            "tests/test_module.py",
        ]
        assert calls[4] == ["poetry", "run", "pytest", "tests/test_module.py"]

    def test_run_scoped_skips_when_no_python_files(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_scoped() skips QC when no Python files changed."""
        calls: list[list[str]] = []

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            result = Mock()
            result.returncode = 0
            result.stdout = " M README.md\n M config.yaml\n"
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        runner.run_scoped()

        # Should only call git status, skip all QC tools
        assert len(calls) == 1
        assert calls[0] == ["git", "status", "--porcelain"]

    def test_run_scoped_skips_tests_when_no_test_files(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_scoped() skips pytest when no test files changed."""
        calls: list[list[str]] = []

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            result = Mock()
            result.returncode = 0
            result.stdout = " M src/module.py\n"
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        runner.run_scoped()

        # Should run black, ruff, pyright but not pytest
        assert len(calls) == 4
        assert any("black" in call for call in calls)
        assert any("ruff" in call for call in calls)
        assert any("pyright" in call for call in calls)
        assert not any("pytest" in call for call in calls)

    def test_run_scoped_raises_on_tool_failure(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_scoped() raises CalledProcessError when tool fails."""

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "git" in argv:
                result = Mock()
                result.returncode = 0
                result.stdout = " M src/module.py\n"
                return result  # type: ignore[return-value]
            # Fail on black
            raise subprocess.CalledProcessError(1, argv)

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        with pytest.raises(subprocess.CalledProcessError):
            runner.run_scoped()


class TestQCRunnerRunFull:
    """Tests for run_full() method."""

    def test_run_full_runs_all_tools_on_entire_codebase(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_full() runs Black, Ruff, Pyright, Pytest with full coverage."""
        calls: list[list[str]] = []

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            result = Mock()
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        runner.run_full()

        # Should have called black, ruff, pyright, pytest
        assert len(calls) == 4
        assert calls[0] == ["poetry", "run", "black", "--check", "."]
        assert calls[1] == ["poetry", "run", "ruff", "check"]
        assert calls[2] == ["poetry", "run", "pyright"]
        assert calls[3] == [
            "poetry",
            "run",
            "pytest",
            "--cov=src/lexile_corpus_tuner",
            "--cov-report=xml",
            "--cov-report=term-missing",
        ]

    def test_run_full_raises_on_tool_failure(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_full() raises CalledProcessError when tool fails."""
        call_count = 0

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on ruff
                raise subprocess.CalledProcessError(1, argv)
            result = Mock()
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        with pytest.raises(subprocess.CalledProcessError):
            runner.run_full()


class TestQCRunnerEdgeCases:
    """Edge case tests for QCRunner."""

    def test_run_helper_passes_cwd_to_subprocess(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """_run() passes workspace as cwd to subprocess.run()."""
        captured_kwargs: dict[str, object] = {}

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            captured_kwargs.update(kwargs)  # type: ignore[arg-type]
            result = Mock()
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        runner._run(["echo", "test"])  # pyright: ignore[reportPrivateUsage]

        assert captured_kwargs["cwd"] == tmp_path
        assert captured_kwargs["check"] is True

    def test_run_helper_handles_capture_output_flag(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """_run() passes capture_output flag to subprocess.run()."""
        captured_kwargs: dict[str, object] = {}

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            captured_kwargs.update(kwargs)  # type: ignore[arg-type]
            result = Mock()
            result.returncode = 0
            result.stdout = "output"
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        result = runner._run(
            ["echo", "test"], capture_output=True
        )  # pyright: ignore[reportPrivateUsage]

        assert captured_kwargs["capture_output"] is True
        assert result.stdout == "output"

    def test_changed_files_with_spaces_in_paths(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """changed_files() handles file paths with spaces."""
        git_output = ' M "src/my module.py"\n M tests/test_file.py\n'

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = git_output
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)

        runner = QCRunner(tmp_path)
        files = runner.changed_files()
        # Git --porcelain output quotes paths with spaces
        assert '"src/my module.py"' in files
        assert "tests/test_file.py" in files

"""
QC toolchain execution for atomic task verification.

Supports both scoped QC (changed files only, fast task gate) and full QC
(entire codebase, phase gate).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class QCRunner:
    """
    Execute scoped and full QC toolchains.

    Purpose:
        Runs Black, Ruff, Pyright, and Pytest on changed files (task gate) or
        the entire codebase (phase gate) to enforce quality standards.

    Usage:
        runner = QCRunner(workspace)
        runner.run_scoped()   # After each task
        runner.run_full()     # After each phase

    Flow:
        Scoped QC:
          1. Detect changed Python files via git status
          2. Run Black/Ruff/Pyright on those files only
          3. Run Pytest only on changed test files (fast path)

        Full QC:
          1. Run Black/Ruff/Pyright on entire codebase
          2. Run Pytest with full coverage reporting

    Invariants:
        - workspace must be a git repository
        - Poetry environment must be active and have required tools

    Side Effects:
        - Calls subprocess commands (git, poetry, black, ruff, pyright, pytest)
        - Raises CalledProcessError if any QC step fails
    """

    # Full toolchain commands for phase gates
    FULL_FMT = ["poetry", "run", "black", "--check", "."]
    FULL_LINT = ["poetry", "run", "ruff", "check"]
    FULL_TYPE = ["poetry", "run", "pyright"]
    FULL_TEST = [
        "poetry",
        "run",
        "pytest",
        "--cov=src/lexile_corpus_tuner",
        "--cov-report=xml",
        "--cov-report=term-missing",
    ]

    def __init__(self, workspace: Path) -> None:
        """
        Initialize the QC runner with workspace path.

        Args:
            workspace (Path): Repository root directory.
        """
        self.workspace = workspace

    def run_scoped(self) -> None:
        """
        Run toolchain on changed files only (task gate).

        Purpose:
            Fast QC verification after a single task completes.

        Raises:
            CalledProcessError: If any QC command fails.

        Side Effects:
            Runs git status, black, ruff, pyright, pytest on changed files.
        """
        files = self.changed_files()
        py_files = self._filter_python_files(files)
        test_files = self._filter_test_files(files)

        # No-op if no Python changes
        if not py_files and not test_files:
            return

        # Run formatter, linter, type checker on changed Python files
        if py_files:
            self._run(["poetry", "run", "black", "--check", *py_files])
            self._run(["poetry", "run", "ruff", "check", *py_files])
            self._run(["poetry", "run", "pyright", *py_files])

        # Run tests only for changed test files (fast path)
        if test_files:
            self._run(["poetry", "run", "pytest", *test_files])

    def run_full(self) -> None:
        """
        Run full toolchain on entire codebase (phase gate).

        Purpose:
            Comprehensive QC verification after a phase completes.

        Raises:
            CalledProcessError: If any QC command fails.

        Side Effects:
            Runs black, ruff, pyright, pytest with full coverage.
        """
        self._run(self.FULL_FMT)
        self._run(self.FULL_LINT)
        self._run(self.FULL_TYPE)
        self._run(self.FULL_TEST)

    def changed_files(self) -> list[str]:
        """
        Detect changed files via git status.

        Purpose:
            Identify files modified/added/deleted for scoped QC.

        Returns:
            list[str]: Relative paths of changed files.

        Side Effects:
            Calls git status --porcelain.
        """
        result = self._run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        files: list[str] = []
        for line in result.stdout.splitlines():
            # Format: XY <path>
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
        return files

    def _filter_python_files(self, paths: Iterable[str]) -> list[str]:
        """Filter to .py files only."""
        return [p for p in paths if p.endswith(".py")]

    def _filter_test_files(self, paths: Iterable[str]) -> list[str]:
        """Filter to test files only (tests/ directory)."""
        return [
            p
            for p in paths
            if (p.startswith("tests/") or "/tests/" in p) and p.endswith(".py")
        ]

    def _run(
        self,
        argv: list[str],
        *,
        capture_output: bool = False,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a subprocess command with consistent settings.

        Args:
            argv (list[str]): Command and arguments to execute.
            capture_output (bool): Whether to capture stdout/stderr.
            text (bool): Whether to decode output as text.

        Returns:
            CompletedProcess: Result of subprocess execution.

        Raises:
            CalledProcessError: If command exits with non-zero status.
        """
        return subprocess.run(  # noqa: S603 - argv constructed from trusted constants
            argv,
            cwd=self.workspace,
            check=True,
            capture_output=capture_output,
            text=text,
        )

"""
QC toolchain execution for atomic task verification.

Supports both scoped QC (changed files only, fast task gate) and full QC
(entire codebase, phase gate).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
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

    AUTO_QC_STEP_ORDER = ["black", "ruff", "pyright", "pytest"]
    EXECUTOR_LOCK_BYPASS_ENV = "ATOMIC_EXECUTOR_SKIP_LOCK"

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
            self._run(
                ["poetry", "run", "pytest", *test_files],
                env=self._merge_env(self._lock_bypass_env()),
            )

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
        self._run(
            self.FULL_TEST,
            env=self._merge_env(self._lock_bypass_env()),
        )

    def run_full_loop_with_artifacts(
        self,
        *,
        artifact_paths: dict[str, Path],
        max_loops: int | None = 10,
    ) -> QCLoopResult:
        """
        Run the full QC toolchain loop and capture outputs to artifact files.

        Purpose:
            Execute Black/Ruff/Pyright/Pytest in the required order, restarting
            the loop when formatting changes occur, and record each step's output
            to the specified artifact file.

        Args:
            artifact_paths (dict[str, Path]): Map from step name to output file.
            max_loops (int | None): Maximum loop iterations before aborting.

        Returns:
            QCLoopResult: Success flag and failure details if applicable.

        Raises:
            RuntimeError: If loop iteration exceeds max_loops.

        Side Effects:
            Runs subprocess commands and writes output files under artifacts/.
        """
        loop_count = 0

        # Repeat the toolchain until it completes without formatting changes.
        while True:
            loop_count += 1
            if max_loops is not None and loop_count > max_loops:
                raise RuntimeError(
                    "QC loop exceeded maximum iterations " f"({max_loops})."
                )

            # Capture the current diff signature so we can detect Black changes
            # even when the working tree already has edits.
            before_black = self._diff_signature(exclude_paths=artifact_paths.values())

            # Run Black in write mode and restart the loop if files changed.
            black_result = self._run_and_record(
                argv=["poetry", "run", "black", "."],
                output_path=artifact_paths["black"],
            )
            # Black failure must be fixed before continuing to other steps.
            if black_result.returncode != 0:
                return QCLoopResult(
                    success=False,
                    failure=QCLoopFailure(
                        step="black",
                        returncode=black_result.returncode,
                        output=black_result.output,
                    ),
                    loop_count=loop_count,
                )

            # Detect formatting changes by comparing diffs before/after Black.
            after_black = self._diff_signature(exclude_paths=artifact_paths.values())
            if after_black != before_black:
                continue

            # Run Ruff, Pyright, and Pytest in order.
            ruff_result = self._run_and_record(
                argv=["poetry", "run", "ruff", "check"],
                output_path=artifact_paths["ruff"],
            )
            # Fail fast if linting fails.
            if ruff_result.returncode != 0:
                return QCLoopResult(
                    success=False,
                    failure=QCLoopFailure(
                        step="ruff",
                        returncode=ruff_result.returncode,
                        output=ruff_result.output,
                    ),
                    loop_count=loop_count,
                )

            pyright_result = self._run_and_record(
                argv=["poetry", "run", "pyright"],
                output_path=artifact_paths["pyright"],
            )
            # Type-checking failures should be fixed before testing.
            if pyright_result.returncode != 0:
                return QCLoopResult(
                    success=False,
                    failure=QCLoopFailure(
                        step="pyright",
                        returncode=pyright_result.returncode,
                        output=pyright_result.output,
                    ),
                    loop_count=loop_count,
                )

            pytest_result = self._run_and_record(
                argv=[
                    "poetry",
                    "run",
                    "pytest",
                    "--cov=src/lexile_corpus_tuner",
                    "--cov=scripts/dev_tools",
                    "--cov-report=term-missing",
                ],
                output_path=artifact_paths["pytest"],
                env=self._merge_env(self._lock_bypass_env()),
            )
            # Test failures must be fixed before the loop can complete.
            if pytest_result.returncode != 0:
                return QCLoopResult(
                    success=False,
                    failure=QCLoopFailure(
                        step="pytest",
                        returncode=pytest_result.returncode,
                        output=pytest_result.output,
                    ),
                    loop_count=loop_count,
                )

            return QCLoopResult(success=True, failure=None, loop_count=loop_count)

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
        # Extract the path column so scoped QC targets only changed files.
        for line in result.stdout.splitlines():
            # Format: XY <path>
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
        return files

    def _git_has_changes(self, *, exclude_paths: Iterable[Path] | None = None) -> bool:
        """
        Check if the git working tree has uncommitted changes.

        Purpose:
            Detect formatter modifications so the QC loop can restart from the
            beginning after Black rewrites files.

        Args:
            exclude_paths (Iterable[Path] | None): Paths to ignore when
                determining whether changes occurred. Useful when QC writes
                artifacts that should not trigger a retry loop.

        Returns:
            bool: True if there are uncommitted changes beyond exclusions,
                False otherwise.
        """
        result = self._run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )

        excluded = self._normalize_excluded_paths(exclude_paths)

        # Scan git status output and ignore excluded paths when requested.
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            changed_path = parts[1]
            if changed_path in excluded:
                continue
            return True

        return False

    def _diff_signature(
        self, *, exclude_paths: Iterable[Path] | None = None
    ) -> tuple[tuple[str, str, str], ...]:
        """
        Build a diff signature for the working tree.

        Purpose:
            Capture a stable fingerprint of current diffs so the QC loop can
            determine whether Black introduced changes even when files were
            already modified.

        Args:
            exclude_paths (Iterable[Path] | None): Paths to ignore when building
                the signature.

        Returns:
            tuple[tuple[str, str, str], ...]: Sorted tuples of
                (path, additions, deletions) for each changed file.
        """
        result = self._run(["git", "diff", "--numstat"], capture_output=True, text=True)
        excluded = self._normalize_excluded_paths(exclude_paths)

        signature: list[tuple[str, str, str]] = []
        # Capture line-level change counts per file for a stable diff fingerprint.
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            additions, deletions, path = parts[0], parts[1], parts[2]
            if " => " in path:
                path = path.split(" => ")[-1].strip()
            if path in excluded:
                continue
            signature.append((path, additions, deletions))

        return tuple(sorted(signature))

    def _normalize_excluded_paths(
        self, exclude_paths: Iterable[Path] | None
    ) -> set[str]:
        """
        Normalize excluded paths to repo-relative POSIX strings.

        Purpose:
            Ensure consistent comparisons between path objects and git output.

        Args:
            exclude_paths (Iterable[Path] | None): Paths to normalize.

        Returns:
            set[str]: Normalized paths in POSIX form.
        """
        excluded: set[str] = set()
        if exclude_paths is None:
            return excluded

        # Normalize paths to repo-relative strings for comparison.
        for path in exclude_paths:
            try:
                rel_path = path.relative_to(self.workspace)
            except ValueError:
                rel_path = path
            excluded.add(rel_path.as_posix())

        return excluded

    def _lock_bypass_env(self) -> dict[str, str]:
        """
        Provide an env flag that bypasses the executor lock in tests.

        Purpose:
            The executor holds a lock file during execute-all runs. When QC
            runs pytest in-process, tests that call the executor should not
            fail due to the existing lock, so we set a bypass env var for
            the pytest subprocess only.

        Returns:
            dict[str, str]: Environment overrides enabling lock bypass.
        """
        return {self.EXECUTOR_LOCK_BYPASS_ENV: "1"}

    def _merge_env(self, extra_env: dict[str, str] | None) -> dict[str, str] | None:
        """
        Merge extra environment variables into the current process env.

        Purpose:
            Ensure subprocesses inherit the current environment plus any
            explicit overrides.

        Args:
            extra_env (dict[str, str] | None): Environment overrides to apply.

        Returns:
            dict[str, str] | None: Merged environment or None if no overrides.
        """
        if not extra_env:
            return None

        env = dict(os.environ)
        env.update(extra_env)
        return env

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
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a subprocess command with consistent settings.

        Args:
            argv (list[str]): Command and arguments to execute.
            capture_output (bool): Whether to capture stdout/stderr.
            text (bool): Whether to decode output as text.
            env (dict[str, str] | None): Environment overrides for the command.

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
            env=env,
        )

    def _run_and_record(
        self,
        *,
        argv: list[str],
        output_path: Path,
        env: dict[str, str] | None = None,
    ) -> QCToolResult:
        """
        Execute a command, capture output, and write it to a file.

        Purpose:
            Run a QC step with output capture for artifact logging.

        Args:
            argv (list[str]): Command and arguments to execute.
            output_path (Path): File path for captured output.
            env (dict[str, str] | None): Environment overrides for the command.

        Returns:
            QCToolResult: Captured output and exit status.

        Side Effects:
            Creates parent directories and writes output to disk.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(  # noqa: S603 - argv constructed from trusted constants
            argv,
            cwd=self.workspace,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        output = (result.stdout or "") + (result.stderr or "")
        header = " ".join(argv)
        output_path.write_text(
            f"$ {header}\n(exit {result.returncode})\n\n{output}",
            encoding="utf-8",
        )

        return QCToolResult(
            step=argv[2] if len(argv) > 2 else "command",
            returncode=result.returncode,
            output=output,
        )


@dataclass(frozen=True)
class QCToolResult:
    """
    Captured output and exit status for a QC step.

    Attributes:
        step (str): Step name or command identifier.
        returncode (int): Process exit code.
        output (str): Combined stdout/stderr text.
    """

    step: str
    returncode: int
    output: str


@dataclass(frozen=True)
class QCLoopFailure:
    """
    Failure details for a QC loop.

    Attributes:
        step (str): Step that failed.
        returncode (int): Process exit code.
        output (str): Captured output from the failed step.
    """

    step: str
    returncode: int
    output: str


@dataclass(frozen=True)
class QCLoopResult:
    """
    Result for a QC loop execution.

    Attributes:
        success (bool): True if all steps passed.
        failure (QCLoopFailure | None): Failure details when success is False.
        loop_count (int): Number of iterations executed.
    """

    success: bool
    failure: QCLoopFailure | None
    loop_count: int

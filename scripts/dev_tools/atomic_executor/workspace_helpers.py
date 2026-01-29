"""
Workspace validation and locking helpers for atomic executor.

Purpose:
    Provides workspace resolution, single-run locking, git tree validation,
    and protected branch checks.

Usage:
    from scripts.dev_tools.atomic_executor.workspace_helpers import (
        resolve_workspace,
        acquire_executor_lock,
        release_executor_lock,
        ensure_clean_tree,
        refuse_protected_branch,
    )
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

# Constants for lock file and protected branches
EXECUTOR_LOCK_FILE = ".agent_logs/executor.lock"
EXECUTOR_LOCK_BYPASS_ENV = "ATOMIC_EXECUTOR_SKIP_LOCK"
PROTECTED_BRANCHES = {"main", "master", "development"}


def resolve_workspace(workspace_arg: str | None) -> Path:
    """
    Resolve workspace root directory.

    Purpose:
        Determines the repository root either from explicit CLI argument
        or by inferring from this file's location.

    Args:
        workspace_arg (str | None): Explicit workspace path from CLI,
            or None to auto-detect.

    Returns:
        Path: Resolved workspace root directory.
    """
    if workspace_arg:
        return Path(workspace_arg).resolve()

    # Infer: assume this file lives at <repo>/scripts/dev_tools/atomic_executor/
    return Path(__file__).resolve().parents[3]


def acquire_executor_lock(workspace: Path) -> Path:
    """
    Acquire the single-run lock to prevent concurrent executor sessions.

    Purpose:
        Ensures only one execute-all run is active at a time so that
        `--continue` does not resume unrelated Copilot sessions.
        Allows pytest subprocesses launched by the executor to bypass the
        lock when ATOMIC_EXECUTOR_SKIP_LOCK is set or when pytest exports
        PYTEST_CURRENT_TEST.

    Args:
        workspace (Path): Repository root used to resolve the lock file path.

    Returns:
        Path: The resolved lock file path.

    Raises:
        RuntimeError: If the lock file already exists and cannot be bypassed.

    Side Effects:
        Creates lock file at workspace/.agent_logs/executor.lock.
    """
    lock_path = workspace / EXECUTOR_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Allow pytest subprocesses spawned by the executor to bypass the lock so
    # execute-all tests do not fail when the parent process holds the lock.
    if os.getenv(EXECUTOR_LOCK_BYPASS_ENV) == "1" or os.getenv("PYTEST_CURRENT_TEST"):
        return lock_path

    if lock_path.exists():
        raise RuntimeError(
            f"Atomic executor lock already exists: {lock_path.as_posix()}"
        )

    lock_path.write_text("atomic_executor_lock\n", encoding="utf-8")
    return lock_path


def release_executor_lock(lock_path: Path) -> None:
    """
    Release the single-run lock file if it exists.

    Purpose:
        Cleans up the lock file to allow subsequent executor runs.

    Args:
        lock_path (Path): Path to the lock file to remove.

    Side Effects:
        Removes lock file from filesystem if it exists.
    """
    with contextlib.suppress(FileNotFoundError):
        if lock_path.exists():
            lock_path.unlink()


def ensure_clean_tree(workspace: Path) -> None:
    """
    Verify working tree is clean (no uncommitted changes).

    Purpose:
        Prevents executor runs on dirty working trees to ensure
        reproducibility and avoid losing uncommitted work.

    Args:
        workspace (Path): Repository root.

    Raises:
        RuntimeError: If working tree has uncommitted changes.
        FileNotFoundError: If git executable not found on PATH.

    Side Effects:
        Executes `git status --porcelain` to check working tree state.
    """
    # Use shutil.which for cross-platform git resolution (avoids hardcoding
    # /usr/bin/git or C:\Program Files\Git\bin\git.exe).
    git_exe = shutil.which("git")
    if not git_exe:
        raise FileNotFoundError("Required executable not found on PATH: git")

    result = (
        subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [git_exe, "status", "--porcelain"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
    )
    if result.stdout.strip():
        raise RuntimeError("Working tree is not clean. Commit/stash before running.")


def refuse_protected_branch(workspace: Path) -> None:
    """
    Refuse execution on protected branches.

    Purpose:
        Prevents atomic executor from running on main/master/development
        branches to avoid accidental commits to protected refs.

    Args:
        workspace (Path): Repository root.

    Raises:
        RuntimeError: If current branch is in PROTECTED_BRANCHES set.

    Side Effects:
        Executes `git rev-parse --abbrev-ref HEAD` to determine current branch.
    """
    branch = _current_branch(workspace)
    if branch and branch in PROTECTED_BRANCHES:
        raise RuntimeError(f"Refusing to run on protected branch '{branch}'.")


def _current_branch(workspace: Path) -> str | None:
    """
    Get current git branch name.

    Purpose:
        Detects the current branch for protected branch validation.

    Args:
        workspace (Path): Repository root.

    Returns:
        str | None: Current branch name, or None if git command fails.

    Side Effects:
        Executes `git rev-parse --abbrev-ref HEAD`.
    """
    # Use shutil.which for cross-platform git resolution.
    git_exe = shutil.which("git")
    if not git_exe:
        return None

    try:
        result = subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [git_exe, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        b = result.stdout.strip()
        return b or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

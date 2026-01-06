"""Remove the devcontainer container and its named workspace volumes.

Purpose:
    Provide a deterministic way to reset the devcontainer environment by
    removing the running container (if present) and its associated named
    volumes. This is useful when rebuilding the container and ensuring no
    stale state persists in the workspace volumes.

Usage:
    poetry run python -m scripts.dev_tools.clean_devcontainer

Flow:
    1) Remove the devcontainer container (ignoring "not found" errors).
    2) Remove the workspace and background volumes (ignoring "not found").
    3) Emit a non-zero exit code if any removal fails for reasons other than
       the resource being absent.

Side Effects:
    - Invokes the local Docker CLI (requires Docker to be available).
    - Deletes the specified container and volumes if they exist.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

DEFAULT_CONTAINER_NAME = "lexile-corpus-tuner-devcontainer"
DEFAULT_VOLUMES = (
    "lexile-corpus-tuner-workspace",
    "lexile-corpus-tuner-workspace-bg",
)


@dataclass
class CommandResult:
    """Represents the outcome of a subprocess command.

    Purpose:
        Capture the return code, standard output, and standard error from a
        Docker CLI invocation so that callers can reason about success,
        "not found" cases, and unexpected failures.

    Attributes:
        command: The full command that was executed.
        returncode: The exit code reported by the subprocess.
        stdout: Captured standard output text.
        stderr: Captured standard error text.
    """

    command: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(command: Sequence[str]) -> CommandResult:
    """Execute a command and capture its result.

    Purpose:
        Provide a thin, typed wrapper around ``subprocess.run`` so the rest of
        the module can be mocked in tests without touching the subprocess
        module directly.

    Args:
        command: The command (including arguments) to execute.

    Returns:
        CommandResult: Structured output from the process invocation.

    Raises:
        OSError: If the process cannot be spawned (e.g., Docker CLI missing).

    Side Effects:
        Spawns a subprocess and consumes its stdout/stderr streams.
    """

    completed = subprocess.run(  # noqa: S603
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandResult(
        command=list(command),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _is_not_found(result: CommandResult) -> bool:
    """Determine whether Docker reported a missing resource.

    Args:
        result: The command result to inspect.

    Returns:
        True if stderr mentions the resource is missing; otherwise False.
    """

    lowered = (result.stderr or "").lower()
    return "no such" in lowered or "not found" in lowered


def _remove_resource(
    command: Sequence[str], runner: Callable[[Sequence[str]], CommandResult]
) -> bool:
    """Run a Docker removal command and interpret its outcome.

    Purpose:
        Treat missing resources as success and only fail on unexpected errors
        (e.g., permission issues or daemon not reachable).

    Args:
        command: Docker CLI command to execute.
        runner: Callable that executes the command and returns CommandResult.

    Returns:
        True when the removal succeeded or the resource was already absent;
        False when the command failed for another reason.
    """

    result: CommandResult = runner(command)
    if result.returncode == 0:
        return True

    if _is_not_found(result):
        return True

    return False


def clean_devcontainer(
    *,
    container_name: str = DEFAULT_CONTAINER_NAME,
    volumes: Iterable[str] = DEFAULT_VOLUMES,
    runner: Callable[[Sequence[str]], CommandResult] = run_command,
) -> int:
    """Remove the devcontainer container and its volumes.

    Args:
        container_name: Name of the devcontainer container to remove.
        volumes: Iterable of volume names to delete.
        runner: Command executor, injected for testability.

    Returns:
        Exit code (0 = success, 1 = failure on any resource).

    Side Effects:
        Invokes Docker CLI commands and may delete containers/volumes.
    """

    failed = False

    # Remove the devcontainer container if present.
    if not _remove_resource(["docker", "rm", "-f", container_name], runner):
        failed = True

    # Remove each volume, treating missing volumes as success.
    for volume in volumes:
        # Iterate over volumes sequentially so failures are captured individually.
        if not _remove_resource(["docker", "volume", "rm", volume], runner):
            failed = True

    return 0 if not failed else 1


def main() -> int:
    """Entry point for CLI usage."""

    return clean_devcontainer()


if __name__ == "__main__":
    sys.exit(main())

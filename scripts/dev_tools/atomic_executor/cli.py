"""
CLI entry point and orchestration for atomic executor.

Provides argument parsing, workspace validation, and main execution loop
that coordinates PlanParser, FeatureResolver, QCRunner, and PromptBuilder.
"""

from __future__ import annotations

import re
import signal
from pathlib import (
    Path,  # - Path required at runtime for variable annotations
)

from scripts.dev_tools.atomic_executor.plan_parser import (
    PlanParser,
    PlanTask,
)
from scripts.dev_tools.atomic_executor.workspace_helpers import (
    release_executor_lock,
)

LOG_DIR = ".agent_logs"

# When Copilot CLI cannot request approval (common in headless/non-interactive
# runs), it emits this exact substring and may then stall until an idle-timeout.
# We detect it during output streaming and fail fast with actionable guidance.
COPILOT_PERMISSION_DENIED_SUBSTRING = (
    "Permission denied and could not request permission from user"
)

# Graceful shutdown state: set by signal handler to request termination.
_shutdown_requested = False
_active_lock_path: Path | None = None


def handle_shutdown_signal(signum: int, frame: object) -> None:
    """
    Signal handler for graceful shutdown (SIGINT/SIGTERM).

    Purpose:
        Sets the global shutdown flag so the main loop can exit cleanly
        after the current task, and releases the lock file immediately
        to avoid leaving stale locks.

    Args:
        signum: Signal number received.
        frame: Current stack frame (unused).
    """
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else signum
    print(f"\n[atomic_executor] Received {sig_name}, shutting down gracefully...")
    # Release lock immediately to avoid stale locks on forced termination
    if _active_lock_path is not None:
        release_executor_lock(_active_lock_path)


def is_shutdown_requested() -> bool:
    """Check if graceful shutdown has been requested via signal."""
    return _shutdown_requested


class CopilotPermissionDeniedError(RuntimeError):
    """Raised when Copilot output indicates an approval/permission dead-end.

    Purpose:
        Provides a typed signal that the Copilot CLI emitted the known
        permission-denied substring and is unlikely to recover without
        additional permissions or an interactive approval path.
    """


READ_TASK_PATTERN = re.compile(r"^read\b", re.IGNORECASE)


def is_phase0_read_task(task: PlanTask) -> bool:
    """
    Determine whether a task is a Phase 0 "read" task.

    Purpose:
        Allows the executor to bundle Phase 0 read tasks into the first prompt.

    Args:
        task (PlanTask): Task to evaluate.

    Returns:
        bool: True when the task is Phase 0 and begins with "read".
    """
    if task.phase != 0:
        return False
    return bool(READ_TASK_PATTERN.match(task.title.strip()))


def phase0_read_tasks(parser: PlanParser) -> list[PlanTask]:
    """
    Collect unchecked Phase 0 read tasks in plan order.

    Purpose:
        Provides the executor and prompt builder a deterministic list of
        tasks to bundle into the first prompt.

    Args:
        parser (PlanParser): Parsed plan access.

    Returns:
        list[PlanTask]: Unchecked Phase 0 read tasks.
    """
    plan = parser.parse()
    read_tasks: list[PlanTask] = []

    # Preserve phase/task ordering for deterministic prompt sequencing.
    for task in sorted(plan.tasks, key=lambda x: (x.phase, x.task_num)):
        if task.checked:
            continue
        if is_phase0_read_task(task):
            read_tasks.append(task)

    return read_tasks


def first_non_read_task(parser: PlanParser) -> PlanTask | None:
    """
    Return the first unchecked task that is not a Phase 0 read task.

    Purpose:
        Ensures the first prompt can combine Phase 0 reads with the first
        actionable non-read task.

    Args:
        parser (PlanParser): Parsed plan access.

    Returns:
        PlanTask | None: First non-read task, or None if none exist.
    """
    plan = parser.parse()

    # Scan tasks in order while skipping Phase 0 read tasks.
    for task in sorted(plan.tasks, key=lambda x: (x.phase, x.task_num)):
        if task.checked:
            continue
        if is_phase0_read_task(task):
            continue
        return task

    return None


# Main entry point now in task_execution module
from scripts.dev_tools.atomic_executor.task_execution import (  # noqa: E402
    main,
)

if __name__ == "__main__":
    main()

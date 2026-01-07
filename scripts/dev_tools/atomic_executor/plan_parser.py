"""
Plan parsing and manipulation for atomic execution.

Handles parsing plan.md files with [P#-T#] task syntax, extracting phase/task
metadata, and flipping checkboxes when tasks complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

TASK_LINE_RE = re.compile(
    r"""
    ^(?P<indent>\s*)-\s*\[(?P<state>[ xX])\]\s*
    \[(?P<task_id>P(?P<phase>\d+)-T(?P<task>\d+))\]
    (?P<title>.*)$
    """,
    re.VERBOSE,
)

PHASE_HEADING_RE = re.compile(r"^\s*#+\s*Phase\s+(?P<phase>\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PlanTask:
    """
    A single task within an atomic execution plan.

    Purpose:
        Represents one executable unit with phase/task numbering and completion state.

    Attributes:
        task_id (str): Unique identifier like "P2-T3".
        phase (int): Phase number (0-indexed or 1-indexed per plan).
        task_num (int): Task number within phase.
        title (str): Human-readable task description.
        checked (bool): True if checkbox is marked [x], False if [ ].
        line_index (int): Zero-based line number in plan.md (for editing).
    """

    task_id: str
    phase: int
    task_num: int
    title: str
    checked: bool
    line_index: int


@dataclass(frozen=True)
class PlanModel:
    """
    Parsed representation of an atomic execution plan.

    Purpose:
        Holds all tasks and phase metadata extracted from plan.md.

    Attributes:
        tasks (list[PlanTask]): All tasks found in the plan.
        phases (list[int]): Sorted list of phase numbers present in the plan.
    """

    tasks: list[PlanTask]
    phases: list[int]


class PlanParser:
    """
    Parse and manipulate atomic execution plans.

    Purpose:
        Parses plan.md files with [P#-T#] checkbox syntax and provides operations
        for task lookup, phase completion checking, and checkbox flipping.

    Usage:
        parser = PlanParser(plan_path)
        plan = parser.parse()
        task = parser.next_unchecked_task()
        parser.flip_checkbox(task)

    Flow:
        1. Read plan.md text
        2. Parse task lines via regex
        3. Extract phase numbers from headings
        4. Provide query/mutation operations

    Invariants:
        - plan_path must exist and be readable
        - Mutations (flip_checkbox) rewrite plan.md atomically
        - Task IDs must be unique within a plan

    Side Effects:
        - flip_checkbox() writes to disk (plan.md)
    """

    def __init__(self, plan_path: Path) -> None:
        """
        Initialize the parser with a path to plan.md.

        Args:
            plan_path (Path): Path to the plan.md file to parse.

        Raises:
            FileNotFoundError: If plan_path does not exist.
        """
        if not plan_path.is_file():
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        self.plan_path = plan_path

    def parse(self) -> PlanModel:
        """
        Parse the plan.md file into structured data.

        Purpose:
            Extract all [P#-T#] tasks and phase headings from plan.md.

        Returns:
            PlanModel: Parsed plan with tasks and phases.

        Side Effects:
            Reads from disk (plan.md).
        """
        tasks: list[PlanTask] = []
        phases: set[int] = set()

        plan_text = self._read_text()
        lines = plan_text.splitlines()

        for idx, line in enumerate(lines):
            m = TASK_LINE_RE.match(line)
            if m:
                phase = int(m.group("phase"))
                task_num = int(m.group("task"))
                checked = m.group("state").strip().lower() == "x"
                tasks.append(
                    PlanTask(
                        task_id=m.group("task_id"),
                        phase=phase,
                        task_num=task_num,
                        title=m.group("title").strip(),
                        checked=checked,
                        line_index=idx,
                    )
                )
                phases.add(phase)
                continue

            hm = PHASE_HEADING_RE.match(line)
            if hm:
                phases.add(int(hm.group("phase")))

        return PlanModel(tasks=tasks, phases=sorted(phases))

    def next_unchecked_task(self) -> PlanTask | None:
        """
        Find the first unchecked task in phase/task order.

        Purpose:
            Determine the next task to execute when resuming.

        Returns:
            PlanTask | None: The first unchecked task, or None if all complete.
        """
        plan = self.parse()
        for t in sorted(plan.tasks, key=lambda x: (x.phase, x.task_num)):
            if not t.checked:
                return t
        return None

    def find_task_by_id(self, task_id: str) -> PlanTask:
        """
        Locate a task by its unique identifier.

        Purpose:
            Support --start <task_id> CLI usage.

        Args:
            task_id (str): Task identifier like "P2-T3".

        Returns:
            PlanTask: The matching task.

        Raises:
            RuntimeError: If task_id is not found in the plan.
        """
        plan = self.parse()
        for t in plan.tasks:
            if t.task_id == task_id:
                return t
        raise RuntimeError(f"Task id not found in plan: {task_id}")

    def phase_complete(self, phase: int) -> bool:
        """
        Check if all tasks in a phase are checked.

        Purpose:
            Determine when to trigger full QC (phase gate).

        Args:
            phase (int): Phase number to check.

        Returns:
            bool: True if all tasks in phase are checked, False otherwise.
        """
        plan = self.parse()
        phase_tasks = [t for t in plan.tasks if t.phase == phase]
        return bool(phase_tasks) and all(t.checked for t in phase_tasks)

    def flip_checkbox(self, task_to_check: PlanTask) -> None:
        """
        Mark a task as complete by flipping its checkbox from [ ] to [x].

        Purpose:
            Authoritative edit after task passes QC gates.

        Args:
            task_to_check (PlanTask): The task whose checkbox to flip.

        Raises:
            RuntimeError: If line mismatch detected (safety check).

        Side Effects:
            Writes to disk (plan.md) atomically.
        """
        lines = self._read_text().splitlines()
        line = lines[task_to_check.line_index]

        # Safety check: verify we're editing the correct line
        m = TASK_LINE_RE.match(line)
        if not m or m.group("task_id") != task_to_check.task_id:
            raise RuntimeError(
                "Internal error: plan line mismatch; refusing to edit plan.md."
            )

        if m.group("state").strip().lower() == "x":
            return  # Already checked

        # Replace checkbox state
        new_line = line.replace("- [ ]", "- [x]", 1).replace("- [X]", "- [x]", 1)
        if new_line == line:
            # Fallback: regex replace
            new_line = re.sub(r"^(\s*-\s*)\[\s\]", r"\1[x]", line)

        lines[task_to_check.line_index] = new_line
        self._write_text_atomic("\n".join(lines) + "\n")

    def preflight_validate(self) -> None:
        """
        Validate plan structure before execution starts.

        Purpose:
            Fail fast if plan.md is malformed or missing critical structure.

        Raises:
            RuntimeError: If validation fails (missing Phase 0, no tasks, etc).

        Side Effects:
            Reads from disk (plan.md).
        """
        plan = self.parse()

        # Lightweight MVP preflight: ensure Phase 0 and at least one task exist
        if not any(p == 0 for p in plan.phases):
            raise RuntimeError(
                "BLOCKED at preflight (before [P0-T1]): missing Phase 0 heading."
            )

        if not plan.tasks:
            raise RuntimeError(
                "BLOCKED at preflight (before [P0-T1]): no [P#-T#] "
                "checkbox tasks found."
            )

        # Heuristic: ensure there is some final validation/toolchain phase
        plan_text = self._read_text()
        has_validation_phase = re.search(
            r"Phase\s+\d+.*(validation|release|qa|toolchain|gate)",
            plan_text,
            flags=re.IGNORECASE,
        )
        has_toolchain = re.search(
            r"black.*ruff.*pyright.*pytest",
            plan_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not has_validation_phase and not has_toolchain:
            raise RuntimeError(
                "BLOCKED at preflight (before [P0-T1]): no final validation/"
                "toolchain phase detected (heuristic)."
            )

    def _read_text(self) -> str:
        """Read plan.md text from disk."""
        return self.plan_path.read_text(encoding="utf-8")

    def _write_text_atomic(self, text: str) -> None:
        """Write plan.md atomically via temp file."""
        tmp = self.plan_path.with_suffix(self.plan_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.plan_path)

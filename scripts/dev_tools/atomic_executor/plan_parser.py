"""
Plan parsing and manipulation for atomic execution.

Handles parsing plan.md files with [P#-T#] task syntax, extracting phase/task
metadata, and flipping checkboxes when tasks complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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

QC_STEP_PATTERNS: dict[str, re.Pattern[str]] = {
    "black": re.compile(r"poetry\s+run\s+black\b", re.IGNORECASE),
    "ruff": re.compile(r"poetry\s+run\s+ruff\s+check\b", re.IGNORECASE),
    "pyright": re.compile(r"poetry\s+run\s+pyright\b", re.IGNORECASE),
    "pytest": re.compile(r"poetry\s+run\s+pytest\b", re.IGNORECASE),
}
QC_LOOP_PATTERN = re.compile(
    r"toolchain\s+loop|restart\s+the\s+toolchain", re.IGNORECASE
)


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


@dataclass(frozen=True)
class AutoQCPhase:
    """
    Auto-detected QC phase metadata.

    Purpose:
        Captures the tasks and artifact outputs required for executor-driven
        quality-control loops (Black, Ruff, Pyright, Pytest) without LLM calls.

    Attributes:
        phase (int): Phase number containing the QC tasks.
        task_ids (list[str]): Task identifiers that should be auto-completed.
        step_task_ids (dict[str, str]): Map from QC step name to task id.
        artifact_paths (dict[str, Path]): Output artifact path for each step.
    """

    phase: int
    task_ids: list[str]
    step_task_ids: dict[str, str]
    artifact_paths: dict[str, Path]


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
        self._auto_qc_phases: dict[int, AutoQCPhase] | None = None

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

        # Walk the plan line-by-line to collect tasks and phase headings.
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
        # Select the first unchecked task in phase/task order for deterministic
        # execution sequencing.
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
        # Scan all tasks to find the exact id requested by the CLI.
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
        # Limit to tasks in the requested phase so other phases do not affect
        # completion checks.
        phase_tasks = [t for t in plan.tasks if t.phase == phase]
        return bool(phase_tasks) and all(t.checked for t in phase_tasks)

    def auto_qc_phase_for_task(self, task: PlanTask) -> AutoQCPhase | None:
        """
        Return the auto-QC phase metadata if the task belongs to it.

        Purpose:
            Allows the executor to bypass LLM calls when a task is identified
            as part of an auto-executed QC phase.

        Args:
            task (PlanTask): Task to evaluate.

        Returns:
            AutoQCPhase | None: Phase metadata if task is part of auto QC.
        """
        phases = self._ensure_auto_qc_phases()
        phase = phases.get(task.phase)
        if not phase:
            return None
        return phase if task.task_id in phase.task_ids else None

    def is_auto_qc_phase(self, phase: int) -> bool:
        """
        Check if the given phase was auto-detected as a QC phase.

        Args:
            phase (int): Phase number to check.

        Returns:
            bool: True if this phase is auto-QC, otherwise False.
        """
        return phase in self._ensure_auto_qc_phases()

    def auto_qc_phase_by_number(self, phase: int) -> AutoQCPhase | None:
        """
        Retrieve auto-QC phase metadata by phase number.

        Args:
            phase (int): Phase number to resolve.

        Returns:
            AutoQCPhase | None: Phase metadata if present.
        """
        return self._ensure_auto_qc_phases().get(phase)

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
                "BLOCKED at preflight (before [P0-T1]): no final QA/toolchain "
                "phase detected (heuristic)."
            )

        # Auto-detect QC phases from task contents so executor can run them.
        self._ensure_auto_qc_phases()

    def _ensure_auto_qc_phases(self) -> dict[int, AutoQCPhase]:
        """
        Resolve and cache auto-QC phase metadata.

        Purpose:
            Detect tasks that correspond to quality-control commands and capture
            their artifact outputs to support executor-side toolchain loops.

        Returns:
            dict[int, AutoQCPhase]: Mapping of phase number to metadata.
        """
        if self._auto_qc_phases is None:
            self._auto_qc_phases = self._detect_auto_qc_phases()
        return self._auto_qc_phases

    def _detect_auto_qc_phases(self) -> dict[int, AutoQCPhase]:
        """
        Detect QC phases by scanning task text for toolchain commands.

        Purpose:
            Identify tasks that reference Black/Ruff/Pyright/Pytest commands and
            capture their declared artifact outputs. This is used by the executor
            to auto-run the QC phase without LLM calls.

        Returns:
            dict[int, AutoQCPhase]: Mapping of phase number to metadata.

        Raises:
            RuntimeError: If a detected QC phase is missing required steps or
                artifact outputs.
        """
        plan = self.parse()
        lines = self._read_text().splitlines()
        phases: dict[int, AutoQCPhase] = {}

        # Index tasks by phase and scan their blocks for QC-related cues.
        # Skip Phase 0 entirely: by convention it captures baselines, not QC loops.
        for task in plan.tasks:
            if task.phase == 0:
                continue

            block_lines = self._task_block_lines(lines, task.line_index)
            block_text = "\n".join(block_lines)

            # Determine which QC steps are referenced in this task block.
            matched_steps: list[str] = []
            # Map explicit toolchain commands to step identifiers for the phase.
            for step, pattern in QC_STEP_PATTERNS.items():
                if pattern.search(block_text):
                    matched_steps.append(step)

            # Identify explicit loop tasks without a concrete command.
            is_loop_task = QC_LOOP_PATTERN.search(block_text) is not None

            # Skip tasks that do not reference QC commands or the loop control.
            if not matched_steps and not is_loop_task:
                continue

            phase_meta = phases.get(task.phase)
            # Initialize the phase metadata on first QC-related task.
            if not phase_meta:
                phase_meta = AutoQCPhase(
                    phase=task.phase,
                    task_ids=[],
                    step_task_ids={},
                    artifact_paths={},
                )

            task_ids = [*phase_meta.task_ids, task.task_id]
            step_task_ids = dict(phase_meta.step_task_ids)

            # Record each matched step, ensuring uniqueness within the phase.
            for step in matched_steps:
                if step in step_task_ids:
                    raise RuntimeError(
                        "Auto-QC detection found duplicate step "
                        f"'{step}' in phase {task.phase}."
                    )
                step_task_ids[step] = task.task_id

            phases[task.phase] = AutoQCPhase(
                phase=task.phase,
                task_ids=task_ids,
                step_task_ids=step_task_ids,
                artifact_paths={},  # Auto-generated at runtime, not from plan
            )

        # Validate required steps for any detected QC phase.
        for phase_num, phase_meta in phases.items():
            # Ensure all four core toolchain steps are present.
            required_steps = {"black", "ruff", "pyright", "pytest"}
            missing_steps = sorted(required_steps - set(phase_meta.step_task_ids))
            if missing_steps:
                raise RuntimeError(
                    "Auto-QC detection missing required steps "
                    f"{missing_steps} for phase {phase_num}."
                )

        # Auto-generate artifact paths for each detected QC phase.
        # Uses standard naming: artifacts/qc-{step}.txt
        for phase_num, phase_meta in phases.items():
            auto_paths = {
                step: Path(f"artifacts/qc-{step}.txt")
                for step in phase_meta.step_task_ids
            }
            phases[phase_num] = AutoQCPhase(
                phase=phase_meta.phase,
                task_ids=phase_meta.task_ids,
                step_task_ids=phase_meta.step_task_ids,
                artifact_paths=auto_paths,
            )

        return phases

    def _task_block_lines(self, lines: list[str], start_index: int) -> list[str]:
        """
        Extract the task line and indented continuation lines from the plan.

        Purpose:
            Provide the full task text block so QC detection can search for
            commands and artifact paths across wrapped/indented lines.

        Args:
            lines (list[str]): Plan file lines.
            start_index (int): Index of the task line to expand.

        Returns:
            list[str]: Task line plus any indented continuation lines.
        """
        if start_index < 0 or start_index >= len(lines):
            return []

        task_line = lines[start_index]
        match = TASK_LINE_RE.match(task_line)
        if not match:
            return [task_line]

        task_indent = len(match.group("indent"))
        block_lines = [task_line]

        # Capture indented continuation lines until the next task or section.
        for line in lines[start_index + 1 :]:
            next_match = TASK_LINE_RE.match(line)
            if next_match:
                break

            if not line.strip():
                block_lines.append(line)
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent > task_indent:
                block_lines.append(line)
                continue

            break

        return block_lines

    def _read_text(self) -> str:
        """Read plan.md text from disk."""
        return self.plan_path.read_text(encoding="utf-8")

    def _write_text_atomic(self, text: str) -> None:
        """Write plan.md atomically via temp file."""
        tmp = self.plan_path.with_suffix(self.plan_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.plan_path)

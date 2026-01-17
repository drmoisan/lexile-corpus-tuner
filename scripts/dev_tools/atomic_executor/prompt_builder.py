"""
Prompt construction from templates and feature context.

Builds prompts by combining a template with resolved feature folder context
(plan task excerpt + spec link) and current task metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TCH003 - Path required at runtime for I/O
from typing import TYPE_CHECKING, Protocol

from scripts.dev_tools.atomic_executor.plan_discovery import (
    ResolvedPlan,
    resolve_feature_plan,
)
from scripts.dev_tools.atomic_executor.plan_parser import TASK_LINE_RE

if TYPE_CHECKING:
    from collections.abc import Callable

    from scripts.dev_tools.atomic_executor.plan_parser import PlanTask

LOGGER = logging.getLogger(__name__)


class PromptBuilderFileSystem(Protocol):
    """
    Protocol for filesystem operations used by PromptBuilder.

    Purpose:
        Abstracts file system operations to enable in-memory testing without
        temporary files, satisfying the repo policy that prohibits tmp_path usage.

    Usage:
        Implement this protocol with RealPromptBuilderFileSystem for production
        or InMemoryPromptBuilderFileSystem for tests.

    Invariants:
        - All path arguments are Path objects.
        - read_text raises FileNotFoundError if the path does not exist.
    """

    def is_file(self, path: Path) -> bool:
        """Check if path exists and is a file."""
        ...

    def is_dir(self, path: Path) -> bool:
        """Check if path exists and is a directory."""
        ...

    def read_text(self, path: Path) -> str:
        """Read text content from path. Raises FileNotFoundError if missing."""
        ...

    def glob(self, directory: Path, pattern: str) -> list[Path]:
        """Return sorted list of paths matching pattern under directory."""
        ...


class RealPromptBuilderFileSystem:
    """
    Real filesystem implementation of PromptBuilderFileSystem.

    Purpose:
        Delegates to actual Path methods for production use.

    Usage:
        Used as the default filesystem in PromptBuilder when no fs is injected.
    """

    def is_file(self, path: Path) -> bool:
        """Check if path exists and is a file."""
        return path.is_file()

    def is_dir(self, path: Path) -> bool:
        """Check if path exists and is a directory."""
        return path.is_dir()

    def read_text(self, path: Path) -> str:
        """Read text content from path. Raises FileNotFoundError if missing."""
        return path.read_text(encoding="utf-8")

    def glob(self, directory: Path, pattern: str) -> list[Path]:
        """Return sorted list of paths matching pattern under directory."""
        return sorted(directory.glob(pattern))


class PromptBuilder:
    """
    Build prompts from templates + context.

    Purpose:
        Constructs execution prompts by injecting feature folder context
        (plan task excerpt + spec link) and current task details into a template.

    Usage:
        builder = PromptBuilder(workspace, template_path)
        prompt = builder.build(feature_dir, current_task)

    Flow:
        1. Read template file
        2. Read plan.md and extract current task excerpt
        3. Verify spec.md exists and link to it
        4. Inject resolved context into template
        5. Return complete prompt text

    Invariants:
        - template_path must exist and be readable
        - feature_dir must contain plan.md and spec.md
        - user-story.md is optional (link only)

    Side Effects:
        - Reads from disk (template, plan, spec, story files) via injected fs.

    Attributes:
        workspace (Path): Repository root directory.
        template_path (Path): Path to prompt template file.
        preferred_model (str | None): Preferred AI model name for execution.
        _fs (PromptBuilderFileSystem): Filesystem abstraction for I/O operations.
        _plan_resolver (Callable[[Path], ResolvedPlan]): Function to resolve plan file.
    """

    def __init__(
        self,
        workspace: Path,
        template_path: Path,
        preferred_model: str | None = None,
        fs: PromptBuilderFileSystem | None = None,
        plan_resolver: Callable[[Path], ResolvedPlan] | None = None,
    ) -> None:
        """
        Initialize the prompt builder with template path.

        Args:
            workspace (Path): Repository root directory.
            template_path (Path): Path to prompt template file.
            preferred_model (str | None): Preferred AI model name for execution.
            fs (PromptBuilderFileSystem | None): Filesystem abstraction. Defaults
                to RealPromptBuilderFileSystem for production use.
            plan_resolver (Callable[[Path], ResolvedPlan] | None): Function to
                resolve the plan file from a feature directory. Defaults to
                resolve_feature_plan.

        Raises:
            FileNotFoundError: If template_path does not exist.
        """
        self._fs: PromptBuilderFileSystem = fs or RealPromptBuilderFileSystem()
        self._plan_resolver: Callable[[Path], ResolvedPlan] = (
            plan_resolver or resolve_feature_plan
        )
        if not self._fs.is_file(template_path):
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        self.workspace = workspace
        self.template_path = template_path
        self.preferred_model = preferred_model

    def build(
        self,
        feature_dir: Path,
        current_task: PlanTask,
        retry_context: str | None = None,
    ) -> str:
        """
        Build prompt text from template and feature context.

        Purpose:
            Generate complete prompt for Copilot invocation.

        Args:
            feature_dir (Path): Feature folder containing plan/spec/story files.
            current_task (PlanTask): The task to execute.
            retry_context (str | None): Optional context if this is a retry attempt.

        Returns:
            str: Complete prompt text with injected context.

        Raises:
            FileNotFoundError: If required files (plan.md, spec.md) are missing.

        Side Effects:
            Reads from disk (template, plan, spec, story files) via injected fs.
        """
        template = self._read_text(self.template_path)

        resolved_plan = self._plan_resolver(feature_dir)
        plan_path = resolved_plan.path
        spec_path = feature_dir / "spec.md"
        story_path = feature_dir / "user-story.md"

        if not self._fs.is_file(spec_path):
            raise FileNotFoundError(f"Missing required spec.md: {spec_path}")

        plan_text = self._read_text(plan_path)
        task_block = self._extract_task_block(plan_text, current_task)
        story_path_label = (
            story_path.as_posix() if self._fs.is_file(story_path) else "(missing)"
        )

        retry_section = ""
        if retry_context:
            retry_section = f"""
!!! RETRY ATTEMPT !!!
Your previous attempt failed validation checks. Please address the errors below
before marking the task as complete.

previous_errors:
{retry_context}

!!! END RETRY CONTEXT !!!
"""

        # Model selection instructions
        model_section = ""
        if self.preferred_model:
            model_section = f"""
---- Preferred Model ----

Preferred Model: {self.preferred_model}

This execution uses model "{self.preferred_model}" for task completion.

---- END Preferred Model ----
"""  # noqa: S608 - template text for user instructions, not SQL

        # Basic token replacement for template variables
        agent_name = "GitHub Copilot"

        # Compute feature name as relative path from workspace/docs/features/active/
        # (REQ-004 compliance: use full relative path, not just feature_dir.name)
        active_root = self.workspace / "docs" / "features" / "active"
        try:
            feature_name = feature_dir.relative_to(active_root).as_posix()
        except ValueError:
            # Fall back to just the directory name if not under active root
            feature_name = feature_dir.name

        template = template.replace("<agent>", agent_name).replace(
            "<feature>", feature_name
        )

        # Construct prompt envelope with resolved context
        appended = f"""

Resolved feature folder: {feature_dir.as_posix()}
Feature folder name: {feature_dir.name}

{model_section}CURRENT TASK (execute only this task, do not advance to other tasks):
- [{current_task.task_id}] {current_task.title}
{retry_section}
Constraints:
- Do NOT replan or expand scope. Do not change task order or IDs.
- Make the smallest change set required to complete ONLY the current task.
- You must run the task-step toolchain for changed files, fix any failures,
  and rerun until all checks pass in this Copilot session before you declare
  the task complete:
  If any python -m poetry command fails, retry the same command with python3 -m poetry.
  - python -m poetry run black .
  - python -m poetry run ruff check
  - python -m poetry run pyright
  - python -m poetry run pytest \\
      --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools \\
      --cov-report=term-missing

When you are confident the current task is complete and passes the above
checks, update plan.md by checking ONLY this task:
- Change '- [ ] [{current_task.task_id}]' to '- [x] [{current_task.task_id}]'
- Do not modify other lines.

If `plan.md` does not exist in the feature folder, update this file instead:
- {resolved_plan.update_filename}

Plan file on disk: {resolved_plan.update_filename}
Plan path: {plan_path.as_posix()}
Spec file: {spec_path.as_posix()}
User story file (optional): {story_path_label}

Plan task context:
---- BEGIN plan task ----
{task_block}
---- END plan task ----
"""
        prompt_text = template + appended
        LOGGER.info(
            "Prompt size: %s bytes, %s lines",
            len(prompt_text),
            prompt_text.count(chr(10)),
        )
        if len(prompt_text) > 15_000:
            LOGGER.warning(
                "WARNING: Prompt exceeds target threshold (15KB); "
                "consider reducing context"
            )
        return prompt_text

    def _read_text(self, path: Path) -> str:
        """Read text file via injected filesystem abstraction."""
        return self._fs.read_text(path)

    def _extract_task_block(self, plan_text: str, task: PlanTask) -> str:
        """
        Extract the current task block from plan text.

        Purpose:
            Provide a minimal plan excerpt for the active task without
            expanding the full plan.

        Args:
            plan_text (str): Full contents of the plan file.
            task (PlanTask): Current task metadata from the plan parser.

        Returns:
            str: A task-focused excerpt including the task line and any
                indented continuation lines.
        """
        lines = plan_text.splitlines()
        if not lines:
            return "(plan is empty)"

        start_index: int | None = None
        task_indent = 0

        # Find the task line that matches the current task ID.
        for idx, line in enumerate(lines):
            match = TASK_LINE_RE.match(line)
            if match and match.group("task_id") == task.task_id:
                start_index = idx
                task_indent = len(match.group("indent"))
                break

        if start_index is None:
            return f"(task {task.task_id} not found in plan)"

        block_lines: list[str] = [lines[start_index]]
        include_blank = False

        # Capture continuation lines that are indented under the task line.
        for line in lines[start_index + 1 :]:
            match = TASK_LINE_RE.match(line)
            if match:
                break

            if not line.strip():
                if include_blank:
                    block_lines.append(line)
                continue

            leading = len(line) - len(line.lstrip(" "))
            if leading > task_indent:
                block_lines.append(line)
                include_blank = True
                continue

            # Stop when we reach a non-indented line (new section).
            break

        return "\n".join(block_lines)

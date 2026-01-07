"""
Prompt construction from templates and feature context.

Builds prompts by combining a template with resolved feature folder context
(plan.md, spec.md, user-story.md) and current task metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.dev_tools.atomic_executor.plan_parser import PlanTask


class PromptBuilder:
    """
    Build prompts from templates + context.

    Purpose:
        Constructs execution prompts by injecting feature folder context
        (plan, spec, user story) and current task details into a template.

    Usage:
        builder = PromptBuilder(workspace, template_path)
        prompt = builder.build(feature_dir, current_task)

    Flow:
        1. Read template file
        2. Read plan.md, spec.md, user-story.md (optional)
        3. Inject resolved context into template
        4. Return complete prompt text

    Invariants:
        - template_path must exist and be readable
        - feature_dir must contain plan.md and spec.md
        - user-story.md is optional

    Side Effects:
        - Reads from disk (template, plan, spec, story files)
    """

    def __init__(self, workspace: Path, template_path: Path) -> None:
        """
        Initialize the prompt builder with template path.

        Args:
            workspace (Path): Repository root directory.
            template_path (Path): Path to prompt template file.

        Raises:
            FileNotFoundError: If template_path does not exist.
        """
        if not template_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        self.workspace = workspace
        self.template_path = template_path

    def build(
        self,
        feature_dir: Path,
        current_task: PlanTask,
    ) -> str:
        """
        Build prompt text from template and feature context.

        Purpose:
            Generate complete prompt for Copilot invocation.

        Args:
            feature_dir (Path): Feature folder containing plan/spec/story files.
            current_task (PlanTask): The task to execute.

        Returns:
            str: Complete prompt text with injected context.

        Raises:
            FileNotFoundError: If required files (plan.md, spec.md) missing.

        Side Effects:
            Reads from disk (template, plan, spec, story files).
        """
        template = self._read_text(self.template_path)

        plan_path = feature_dir / "plan.md"
        spec_path = feature_dir / "spec.md"
        story_path = feature_dir / "user-story.md"

        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing required plan.md: {plan_path}")
        if not spec_path.is_file():
            raise FileNotFoundError(f"Missing required spec.md: {spec_path}")

        plan_text = self._read_text(plan_path)
        spec_text = self._read_text(spec_path)
        story_text = self._read_text(story_path) if story_path.is_file() else ""

        # Construct prompt envelope with resolved context
        appended = f"""

Resolved feature folder: {feature_dir.as_posix()}
Feature folder name: {feature_dir.name}

CURRENT TASK (execute only this task, do not advance to other tasks):
- [{current_task.task_id}] {current_task.title}

Constraints:
- Follow all repository policies under .github/instructions/.
- Do NOT replan or expand scope. Do not change task order or IDs.
- Make the smallest change set required to complete ONLY the current task.
- After you believe the task is done, run the task-step toolchain for
  changed files only:
  - poetry run black --check <changed .py files>
  - poetry run ruff check <changed .py files>
  - poetry run pyright <changed .py files>
  - poetry run pytest <changed test files>   (only if tests changed)

When you are confident the current task is complete and passes the above
checks, update plan.md by checking ONLY this task:
- Change '- [ ] [{current_task.task_id}]' to '- [x] [{current_task.task_id}]'
- Do not modify other lines.

---- BEGIN plan.md ----
{plan_text}
---- END plan.md ----

---- BEGIN spec.md ----
{spec_text}
---- END spec.md ----

---- BEGIN user-story.md (optional) ----
{story_text}
---- END user-story.md ----
"""
        return template + appended

    def _read_text(self, path: Path) -> str:
        """Read text file from disk."""
        return path.read_text(encoding="utf-8")

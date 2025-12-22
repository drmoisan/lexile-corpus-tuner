from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .models import FeatureDocExcerpt, section, truncate

if TYPE_CHECKING:
    from collections.abc import Iterable


def parse_section(markdown: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = rf"^##\s+{escaped}\s*\r?\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def completed_plan_tasks(markdown: str, *, limit: int = 10) -> list[str]:
    tasks: list[str] = []
    for line in markdown.splitlines():
        if re.search(r"\[x\]", line, flags=re.IGNORECASE):
            cleaned = re.sub(r"^[-*]\s*\[[xX]\]\s*", "", line).strip()
            tasks.append(cleaned)
        if len(tasks) >= limit:
            break
    return tasks


def extract_issue_references(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(r"(?<!\w)#\d+|\b[A-Z][A-Z0-9]+-\d+\b", text)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in matches:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _resolve_feature_dir(base_dir: Path, feature: str) -> Path | None:
    direct = base_dir / feature
    if direct.exists():
        return direct

    pattern = re.compile(rf"(?:^|[-_]){re.escape(feature)}(?:[-_]|$)")
    strong_matches: list[Path] = []
    weak_matches: list[Path] = []

    for candidate in sorted(base_dir.iterdir()):
        if not candidate.is_dir():
            continue
        name = candidate.name
        if pattern.search(name):
            strong_matches.append(candidate)
        elif feature in name:
            weak_matches.append(candidate)

    if strong_matches:
        return strong_matches[0]
    if weak_matches:
        return weak_matches[0]
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _verification_text(plan_text: str) -> str:
    for heading in ("Verification", "Test Plan"):
        section_text = parse_section(plan_text, heading)
        if section_text:
            return section_text
    return ""


def gather_feature_excerpts(
    root: Path, changed_files: Iterable[str]
) -> list[FeatureDocExcerpt]:
    features: set[str] = set()
    for raw in changed_files:
        parts = Path(raw).parts
        if (
            len(parts) >= 4
            and parts[0] == "docs"
            and parts[1] == "features"
            and parts[2] == "active"
        ):
            features.add(parts[3])

    excerpts: list[FeatureDocExcerpt] = []
    base_dir = root / "docs" / "features" / "active"
    promoted_dir = root / "docs" / "features" / "potential" / "promoted"
    for feature in sorted(features):
        feature_dir: Path | None = _resolve_feature_dir(base_dir, feature)
        promoted_feature_dir: Path | None = _resolve_feature_dir(promoted_dir, feature)
        if feature_dir is None and promoted_feature_dir is None:
            continue

        active_dir = feature_dir or promoted_feature_dir
        if active_dir is None:
            continue

        spec_path = active_dir / "spec.md"
        plan_path = active_dir / "plan.md"
        user_story_path: Path = active_dir / "user-story.md"
        promoted_story_path = (
            promoted_feature_dir / "user-story.md"
            if promoted_feature_dir is not None
            else None
        )

        promoted_story_text = (
            _read_text(promoted_story_path) if promoted_story_path else ""
        )
        if promoted_story_path is not None and not user_story_path.exists():
            user_story_path = promoted_story_path

        user_story_text = _read_text(user_story_path)
        if (
            not user_story_text
            and promoted_story_text
            and promoted_story_path is not None
        ):
            user_story_text = promoted_story_text
            user_story_path = promoted_story_path

        spec_text = _read_text(spec_path)
        plan_text = _read_text(plan_path)

        spec_parts: list[str] = []
        for heading in (
            "Context",
            "Root Cause",
            "Root Cause/Problem",
            "Problem",
            "Proposed Fix",
            "Acceptance Criteria",
            "Constraints & Risks",
            "Behavior",
            "Overview",
        ):
            section_text = parse_section(spec_text, heading)
            if section_text:
                spec_parts.append(f"{heading}: {truncate(section_text)}")

        plan_tasks = completed_plan_tasks(plan_text)
        plan_section = (
            "\n".join(f"- {task}" for task in plan_tasks) if plan_tasks else ""
        )
        verification_text = _verification_text(plan_text)
        verification_block = (
            "Plan verification notes:\n" + truncate(verification_text)
            if verification_text
            else ""
        )

        story_parts: list[str] = []
        story_statements = parse_section(user_story_text, "Story Statement")
        if story_statements:
            story_lines = [
                line.strip("- ")
                for line in story_statements.splitlines()
                if line.strip()
            ]
            if story_lines:
                story_parts.append(
                    "Story Statement:\n"
                    + "\n".join(f"- {line}" for line in story_lines)
                )
        problem_section = parse_section(user_story_text, "Problem / Why")
        if problem_section:
            story_parts.append("Problem / Why:\n" + truncate(problem_section))
        if not story_parts and promoted_story_text:
            promoted_problem = parse_section(promoted_story_text, "Problem / Why")
            if not promoted_problem:
                promoted_problem = parse_section(promoted_story_text, "Summary")
            if promoted_problem:
                story_parts.append("Problem / Why:\n" + truncate(promoted_problem))

        lines: list[str] = [section(f"Feature doc: {feature}")]
        if story_parts:
            lines.append("User story excerpts:\n" + "\n\n".join(story_parts))
        if spec_parts:
            lines.append("Spec excerpts:\n" + "\n\n".join(spec_parts))
        if plan_section:
            lines.append("Plan completed tasks:\n" + plan_section)
        if verification_block:
            lines.append(verification_block)
        if len(lines) == 1:
            lines.append("(no spec/plan/user-story excerpts found)")

        context_files = [
            str(path.relative_to(root))
            for path in (spec_path, plan_path, user_story_path)
            if path.exists()
        ]
        issue_refs = extract_issue_references(
            "\n".join([spec_text, plan_text, user_story_text])
        )
        excerpts.append(
            FeatureDocExcerpt(
                feature=feature,
                excerpt="\n".join(lines),
                issue_refs=issue_refs,
                context_files=sorted(set(context_files)),
            )
        )

    return excerpts

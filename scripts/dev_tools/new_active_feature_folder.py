"""Create an active feature folder from templates and optional potential files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
EXCLUDED_POTENTIAL_NAMES = {"template.md", "README.md"}
PLACEHOLDERS = [
    "<feature-name>",
    "<refactor-name>",
    "<epic-name>",
    "<name>",
    "<bug-name>",
]


@dataclass
class IssueMeta:
    number: str
    author: str
    updated_date: str


@dataclass
class ActiveFolderResult:
    target: Path
    potential_issue_path: Path | None


class FileSystem(Protocol):
    def exists(self, path: Path) -> bool: ...

    def ensure_dir(self, path: Path) -> None: ...

    def copy_file(self, src: Path, dest: Path) -> None: ...

    def copy_tree(self, src: Path, dest: Path) -> None: ...

    def list_files(self, path: Path) -> Iterable[Path]: ...

    def read_text(self, path: Path) -> str: ...

    def write_text(self, path: Path, content: str) -> None: ...

    def move(self, src: Path, dest: Path) -> None: ...


@dataclass
class RealFileSystem(FileSystem):
    def exists(self, path: Path) -> bool:
        return path.exists()

    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def copy_file(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    def copy_tree(self, src: Path, dest: Path) -> None:
        for file_path in src.rglob("*"):
            if file_path.is_dir():
                continue
            relative = file_path.relative_to(src)
            target_path = dest / relative
            self.copy_file(file_path, target_path)

    def list_files(self, path: Path) -> Iterable[Path]:
        if not path.exists():
            return []
        return [p for p in path.iterdir() if p.is_file()]

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def move(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_file():
                dest.unlink()
        src.replace(dest)


def resolve_workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_feature_name(feature_name: str) -> None:
    if not feature_name or not NAME_PATTERN.fullmatch(feature_name):
        raise ValueError(
            f"Aborted: '{feature_name}' is invalid. Use kebab/underscore-case "
            "letters/numbers (e.g., notes-feature or notes_feature)."
        )


def format_checklist(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        trimmed = raw_line.strip()
        if not trimmed:
            continue
        if re.match(r"^-\s*\[?\s*\]", trimmed):
            lines.append(trimmed)
        elif trimmed.startswith("-"):
            lines.append(trimmed)
        else:
            lines.append(f"- [ ] {trimmed}")
    return "\n".join(lines)


def get_section(content: str, name: str) -> str:
    pattern = re.compile(
        rf"^\s*##\s+{re.escape(name)}\s*\r?\n(.*?)(?=^\s*##\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        return ""
    return match.group(1).strip()


def set_section(content: str, name: str, body: str) -> str:
    if not body or not body.strip():
        return content

    pattern = re.compile(
        rf"(^##\s+{re.escape(name)}\s*\r?\n)(.*?)(?=^\s*##\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    replacement = f"\\1{body}\n\n"
    if pattern.search(content):
        return pattern.sub(replacement, content)

    trimmed = content.rstrip()
    if trimmed:
        trimmed += "\n\n"
    return f"{trimmed}## {name}\n{body}\n"


def set_header_placeholder(
    content: str,
    feature_name: str,
    issue_field: str,
    owner_field: str,
    updated_field: str,
) -> str:
    result = content
    for placeholder in PLACEHOLDERS:
        result = result.replace(placeholder, feature_name)
    result = re.sub(r"#`?<id>`?", issue_field, result)
    result = result.replace("<#id or TBD>", issue_field)
    result = result.replace("#<tracking-issue>", issue_field)
    result = re.sub(
        r"^- Owner:\s+(?:name|<name>)",
        f"- Owner: {owner_field}",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^- Date:\s+YYYY-MM-DD", f"- Date: {updated_field}", result, flags=re.MULTILINE
    )
    result = re.sub(
        r"^- Last Updated:\s+YYYY-MM-DD",
        f"- Last Updated: {updated_field}",
        result,
        flags=re.MULTILINE,
    )
    if not re.search(r"^- Issue:\s*#?", result, flags=re.MULTILINE):
        result = f"- Issue: {issue_field}\n{result}"
    return result


def find_potential_file(
    feature_name: str, workspace: Path, fs: FileSystem
) -> Path | None:
    normalized = feature_name.replace("_", "-")
    potential_dirs = [
        workspace / "docs" / "features" / "potential",
        workspace / "docs" / "features" / "potential" / "promoted",
    ]

    for directory in potential_dirs:
        candidates = [
            file
            for file in fs.list_files(directory)
            if file.suffix == ".md"
            and normalized in file.name
            and file.name not in EXCLUDED_POTENTIAL_NAMES
        ]
        if candidates:
            return sorted(candidates, key=lambda path: path.name, reverse=True)[0]
    return None


def parse_issue_number(content: str) -> str | None:
    match = re.search(r"^\s*-\s*Issue\s*:\s*#?(\d+)", content, flags=re.MULTILINE)
    if match:
        return match.group(1)
    return None


def build_folder_slug(
    feature_name: str, potential_file: Path | None, issue_number: str | None
) -> str:
    slug = feature_name.replace("_", "-")
    if potential_file:
        slug = potential_file.stem
    if issue_number and not slug.endswith(str(issue_number)):
        slug = f"{slug}-{issue_number}"
    if not NAME_PATTERN.fullmatch(slug):
        raise ValueError(
            f"Aborted: '{slug}' is invalid. Use kebab/underscore-case letters/numbers "
            "(e.g., notes-feature or notes_feature)."
        )
    return slug


def copy_template(
    feature_type: str, template_dir: Path, target_dir: Path, fs: FileSystem
) -> None:
    if feature_type == "bug":
        for name in ("spec.md", "plan.md"):
            src = template_dir / name
            if fs.exists(src):
                fs.copy_file(src, target_dir / name)
    else:
        fs.copy_tree(template_dir, target_dir)


def default_issue_fetcher(issue_number: str) -> IssueMeta | None:
    gh_cmd = shutil.which("gh")
    if not gh_cmd:
        return None
    result = subprocess.run(  # noqa: S603
        [
            gh_cmd,
            "issue",
            "view",
            issue_number,
            "--json",
            "number,title,url,author,updatedAt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        parsed = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None

    number = str(parsed.get("number", issue_number))
    author = parsed.get("author", {}).get("login") or "name"
    updated_at = parsed.get("updatedAt")
    updated_date = "YYYY-MM-DD"
    if updated_at:
        try:
            updated_date = str(updated_at).split("T")[0]
        except Exception:
            updated_date = "YYYY-MM-DD"
    return IssueMeta(number=number, author=author, updated_date=updated_date)


def default_code_launcher(files: Iterable[Path]) -> bool:
    code_cmd = shutil.which("code")
    if not code_cmd:
        return False
    subprocess.run([code_cmd, *[str(f) for f in files]], check=False)  # noqa: S603
    return True


def _apply_header_and_sections(
    path: Path,
    feature_name: str,
    issue_field: str,
    owner_field: str,
    updated_field: str,
    fs: FileSystem,
    updates: list[tuple[str, str]],
) -> None:
    if not fs.exists(path):
        return
    content = fs.read_text(path)
    content = set_header_placeholder(
        content, feature_name, issue_field, owner_field, updated_field
    )
    for section_name, body in updates:
        content = set_section(content, section_name, body)
    fs.write_text(path, content)


def update_feature_docs(
    feature_type: str,
    feature_name: str,
    target_dir: Path,
    issue_field: str,
    owner_field: str,
    updated_field: str,
    fs: FileSystem,
    sections: dict[str, str],
) -> list[Path]:
    files_to_open: list[Path] = []
    if feature_type == "feature":
        user_story = target_dir / "user-story.md"
        spec = target_dir / "spec.md"
        plan = target_dir / "plan.md"
        _apply_header_and_sections(
            user_story,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            fs,
            [
                ("Problem / Why", sections.get("problem", "")),
                ("Acceptance Criteria", format_checklist(sections.get("criteria", ""))),
            ],
        )
        _apply_header_and_sections(
            spec,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            fs,
            [
                ("Overview", sections.get("problem", "")),
                ("Behavior", sections.get("behavior", "")),
                ("Constraints & Risks", sections.get("constraints", "")),
                (
                    "Seeded Test Conditions (from potential)",
                    format_checklist(sections.get("tests", "")),
                ),
            ],
        )
        _apply_header_and_sections(
            plan, feature_name, issue_field, owner_field, updated_field, fs, []
        )
        files_to_open.extend([user_story, spec, plan])
    elif feature_type == "refactor":
        spec = target_dir / "spec.md"
        plan = target_dir / "plan.md"
        _apply_header_and_sections(
            spec,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            fs,
            [
                ("Intent & Outcomes", sections.get("problem", "")),
                ("Scope (structural changes)", sections.get("behavior", "")),
                ("Risks & Mitigations", sections.get("constraints", "")),
                (
                    "Seeded Test Conditions (from potential)",
                    format_checklist(sections.get("tests", "")),
                ),
            ],
        )
        _apply_header_and_sections(
            plan, feature_name, issue_field, owner_field, updated_field, fs, []
        )
        files_to_open.extend([spec, plan])
    elif feature_type == "epic":
        initiative = target_dir / "initiative.md"
        _apply_header_and_sections(
            initiative, feature_name, issue_field, owner_field, updated_field, fs, []
        )
        files_to_open.append(initiative)
    elif feature_type == "bug":
        spec = target_dir / "spec.md"
        plan = target_dir / "plan.md"

        context_parts: list[str] = []
        if sections.get("bug_summary"):
            context_parts.append(sections["bug_summary"])
        if sections.get("bug_environment"):
            context_parts.append(f"Environment:\n{sections['bug_environment']}")
        if sections.get("bug_impact"):
            context_parts.append(f"Impact / Severity:\n{sections['bug_impact']}")
        context_body = "\n\n".join(context_parts)

        repro_parts: list[str] = []
        if sections.get("bug_steps"):
            repro_parts.append(f"Steps to Reproduce:\n{sections['bug_steps']}")
        expected_actual: list[str] = []
        if sections.get("bug_expected"):
            expected_actual.append(f"Expected:\n{sections['bug_expected']}")
        if sections.get("bug_actual"):
            expected_actual.append(f"Actual:\n{sections['bug_actual']}")
        if expected_actual:
            repro_parts.append("\n\n".join(expected_actual))
        if sections.get("bug_logs"):
            repro_parts.append(f"Logs / Screenshots:\n{sections['bug_logs']}")
        repro_body = "\n\n".join(repro_parts)

        updates: list[tuple[str, str]] = []
        if context_body:
            updates.append(("Context", context_body))
        if repro_body:
            updates.append(("Repro & Evidence", repro_body))
        if sections.get("bug_cause"):
            updates.append(("Root Cause Analysis", sections["bug_cause"]))
        if sections.get("bug_validation"):
            updates.append(("Proposed Fix", sections["bug_validation"]))
            updates.append(("Test Strategy", sections["bug_validation"]))

        _apply_header_and_sections(
            spec,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            fs,
            updates,
        )
        _apply_header_and_sections(
            plan, feature_name, issue_field, owner_field, updated_field, fs, []
        )
        files_to_open.extend([spec, plan])
    return files_to_open


def create_active_folder(
    feature_name: str,
    feature_type: str,
    issue_number: str | None = None,
    force: bool = False,
    workspace: Path | None = None,
    fs: FileSystem | None = None,
    issue_fetcher: Callable[[str], IssueMeta | None] = default_issue_fetcher,
    code_launcher: Callable[[Iterable[Path]], bool] = default_code_launcher,
) -> ActiveFolderResult:
    if feature_type not in {"feature", "refactor", "epic", "bug"}:
        raise ValueError("Type must be one of: feature, refactor, epic, bug")

    validate_feature_name(feature_name)
    workspace_path = workspace or resolve_workspace()
    filesystem = fs or RealFileSystem()

    template_dir = workspace_path / "docs" / "features" / "templates" / feature_type
    if not filesystem.exists(template_dir):
        raise FileNotFoundError(f"Template folder not found: {template_dir}")

    potential_file = find_potential_file(feature_name, workspace_path, filesystem)
    potential_content = filesystem.read_text(potential_file) if potential_file else ""

    normalized_issue_number = (issue_number or "").strip() or None
    if normalized_issue_number and normalized_issue_number.lower() == "auto":
        normalized_issue_number = None
    if not normalized_issue_number:
        normalized_issue_number = parse_issue_number(potential_content)

    folder_slug = build_folder_slug(
        feature_name, potential_file, normalized_issue_number
    )
    target_dir = workspace_path / "docs" / "features" / "active" / folder_slug

    if filesystem.exists(target_dir) and not force:
        raise FileExistsError(
            f"Target exists: {target_dir}. Re-run with --force to overwrite."
        )

    filesystem.ensure_dir(target_dir)
    copy_template(feature_type, template_dir, target_dir, filesystem)

    issue_meta = None
    if normalized_issue_number:
        issue_meta = issue_fetcher(normalized_issue_number)
    issue_field = f"#{normalized_issue_number}" if normalized_issue_number else "#<id>"
    if issue_meta:
        issue_field = f"#{issue_meta.number}"
    owner_field = issue_meta.author if issue_meta else "name"
    updated_field = issue_meta.updated_date if issue_meta else "YYYY-MM-DD"

    sections: dict[str, str] = {
        "problem": get_section(potential_content, "Problem / Why"),
        "behavior": get_section(potential_content, "Proposed Behavior"),
        "criteria": get_section(potential_content, "Acceptance Criteria (early draft)"),
        "constraints": get_section(potential_content, "Constraints & Risks"),
        "tests": get_section(potential_content, "Test Conditions to Consider"),
        "bug_summary": get_section(potential_content, "Summary"),
        "bug_environment": get_section(potential_content, "Environment"),
        "bug_steps": get_section(potential_content, "Steps to Reproduce"),
        "bug_expected": get_section(potential_content, "Expected Behavior"),
        "bug_actual": get_section(potential_content, "Actual Behavior"),
        "bug_logs": get_section(potential_content, "Logs / Screenshots"),
        "bug_impact": get_section(potential_content, "Impact / Severity"),
        "bug_cause": get_section(potential_content, "Suspected Cause / Notes"),
        "bug_validation": get_section(
            potential_content, "Proposed Fix / Validation Ideas"
        ),
    }

    files_to_open = update_feature_docs(
        feature_type,
        feature_name,
        target_dir,
        issue_field,
        owner_field,
        updated_field,
        filesystem,
        sections,
    )

    potential_issue_path = None
    if potential_file:
        potential_issue_path = target_dir / "issue.md"
        filesystem.move(potential_file, potential_issue_path)
        print(f"Moved potential file to {potential_issue_path}")

    if potential_file:
        print(f"Seeded docs from potential: {potential_file.name}")

    if files_to_open:
        existing = [path for path in files_to_open if filesystem.exists(path)]
        if potential_issue_path:
            existing.append(potential_issue_path)
        if existing:
            opened = code_launcher(existing)
            if not opened:
                print("VS Code 'code' command not found. Files to edit:")
                for path in existing:
                    print(f"  {path}")

    print(f"Created/updated: {target_dir}")
    return ActiveFolderResult(
        target=target_dir, potential_issue_path=potential_issue_path
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create docs/features/active/<name>/ from the selected template."
    )
    parser.add_argument(
        "--feature-name", required=True, help="Feature folder name (kebab/underscore)"
    )
    parser.add_argument(
        "--type",
        dest="feature_type",
        choices=["feature", "refactor", "epic", "bug"],
        default="feature",
        help="Type of folder to create",
    )
    parser.add_argument(
        "--issue-number",
        dest="issue_number",
        default=None,
        help="Issue number or 'auto'",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing target"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        create_active_folder(
            feature_name=args.feature_name,
            feature_type=args.feature_type,
            issue_number=args.issue_number,
            force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        print(f"Aborted: required file not found: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

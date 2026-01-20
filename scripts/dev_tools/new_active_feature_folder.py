"""Create an active feature folder from templates and optional potential files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

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

PLAN_TIMESTAMP_TEMPLATE_NAME = "plan.yyyy-MM-ddTHH-mm.md"


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


def get_est_timestamp(now_provider: Callable[[], datetime] | None = None) -> str:
    """Return a Windows-friendly ISO-ish timestamp string for America/New_York.

    Purpose:
        Generate a single timestamp token that can safely be used in filenames and
        inserted into plan documents. The repo conventions use `YYYY-MM-DDTHH-mm`
        (hyphen in the time portion) to avoid `:` which is illegal in Windows
        filenames.

    Args:
        now_provider (Callable[[], datetime] | None): Optional clock injection for
            tests. When provided, its return value is interpreted as a timezone-aware
            datetime.

    Returns:
        str: Timestamp formatted as `YYYY-MM-DDTHH-mm` in America/New_York.

    Raises:
        ValueError: If now_provider returns a naive datetime.
    """

    # Prefer a test-injected clock for determinism; otherwise use local wall time.
    now = (
        now_provider()
        if now_provider
        else datetime.now(tz=ZoneInfo("America/New_York"))
    )
    if now.tzinfo is None:
        raise ValueError("now_provider must return a timezone-aware datetime")
    localized = now.astimezone(ZoneInfo("America/New_York"))
    return localized.strftime("%Y-%m-%dT%H-%M")


def extract_date_from_timestamp(timestamp: str) -> str:
    """Extract `YYYY-MM-DD` date component from a `YYYY-MM-DDTHH-mm` timestamp."""

    return timestamp.split("T", 1)[0]


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
    status_field: str | None = None,
    parent_field: str | None = None,
    version_field: str | None = None,
) -> str:
    """Replace template placeholders in the frontmatter/header block.

    Purpose:
        Fill in issue/owner/timestamp/status/version metadata while keeping the
        templates aligned with the active folder creation flow.

    Args:
        content (str): Raw template content.
        feature_name (str): Slug used to replace name placeholders.
        issue_field (str): Issue identifier (e.g., #73 or TBD).
        owner_field (str): Owner name or placeholder value.
        updated_field (str): Last-updated timestamp token.
        status_field (str | None): Status text to apply when present.
        parent_field (str | None): Parent issue identifier or placeholder value.
        version_field (str | None): Version token to apply when present.

    Returns:
        str: Updated content with frontmatter placeholders replaced.
    """
    result = content
    for placeholder in PLACEHOLDERS:
        result = result.replace(placeholder, feature_name)
    result = result.replace("<issue>", issue_field)
    if parent_field is not None:
        result = result.replace("<parent-id>", parent_field)
    if status_field is not None:
        result = result.replace("<status>", status_field)
    if version_field is not None:
        result = result.replace("<version_number>", version_field)
    result = re.sub(r"#`?<id>`?", issue_field, result)
    result = result.replace("<#id or TBD>", issue_field)
    result = result.replace("#<tracking-issue>", issue_field)

    # Update Issue/Owner/Date/Status fields in both plain and bold-list formats.
    # These variants appear across multiple templates.
    result = re.sub(
        r"^-\s*\*\*Issue:\*\*\s+.*$",
        f"- **Issue:** {issue_field}",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^-\s*Issue\s*:\s+.*$",
        f"- Issue: {issue_field}",
        result,
        flags=re.MULTILINE,
    )

    result = re.sub(
        r"^-\s*\*\*Owner:\*\*\s+(?:name|<name>|.*)$",
        f"- **Owner:** {owner_field}",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^-\s*Owner\s*:\s+(?:name|<name>|.*)$",
        f"- Owner: {owner_field}",
        result,
        flags=re.MULTILINE,
    )

    if parent_field is not None:
        result = re.sub(
            r"^-\s*\*\*Parent \(optional\):\*\*\s+.*$",
            f"- **Parent (optional):** {parent_field}",
            result,
            flags=re.MULTILINE,
        )
        result = re.sub(
            r"^-\s*Parent \(optional\)\s*:\s+.*$",
            f"- Parent (optional): {parent_field}",
            result,
            flags=re.MULTILINE,
        )

    result = re.sub(
        r"^-\s*\*\*Last Updated:\*\*\s+.*$",
        f"- **Last Updated:** {updated_field}",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^-\s*Last Updated\s*:\s+.*$",
        f"- Last Updated: {updated_field}",
        result,
        flags=re.MULTILINE,
    )

    result = re.sub(
        r"^-\s*\*\*Date:\*\*\s+.*$",
        f"- **Date:** {updated_field}",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^-\s*Date\s*:\s+YYYY-MM-DD$",
        f"- Date: {updated_field}",
        result,
        flags=re.MULTILINE,
    )
    result = result.replace("<yyyy-MM-ddTHH-mm>", updated_field)

    if status_field is not None:
        result = re.sub(
            r"^-\s*\*\*Status:\*\*\s+.*$",
            f"- **Status:** {status_field}",
            result,
            flags=re.MULTILINE,
        )
        result = re.sub(
            r"^-\s*Status\s*:\s+.*$",
            f"- Status: {status_field}",
            result,
            flags=re.MULTILINE,
        )

    if version_field is not None:
        result = re.sub(
            r"^-\s*\*\*Version:\*\*\s+.*$",
            f"- **Version:** {version_field}",
            result,
            flags=re.MULTILINE,
        )
        result = re.sub(
            r"^-\s*Version\s*:\s+.*$",
            f"- Version: {version_field}",
            result,
            flags=re.MULTILINE,
        )

    if not re.search(
        r"^-\s*(?:\*\*Issue:\*\*\s*|Issue\s*:)\s*#?",
        result,
        flags=re.MULTILINE,
    ):
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
        for name in ("spec.md", PLAN_TIMESTAMP_TEMPLATE_NAME, "plan.md"):
            src = template_dir / name
            if fs.exists(src):
                fs.copy_file(src, target_dir / name)
                # Prefer the timestamped plan template when both exist.
                if name == PLAN_TIMESTAMP_TEMPLATE_NAME:
                    break
    else:
        fs.copy_tree(template_dir, target_dir)


def materialize_plan_file(
    feature_type: str,
    target_dir: Path,
    feature_name: str,
    issue_field: str,
    owner_field: str,
    parent_field: str,
    status_field: str,
    version_field: str,
    plan_timestamp: str,
    fs: FileSystem,
) -> Path | None:
    """Rename and stamp plan templates when a timestamped plan template exists.

    Purpose:
        Some templates use a timestamp placeholder in the plan filename
        (`plan.yyyy-MM-ddTHH-mm.md`). This function renames that file to
        `plan.<timestamp>.md` and updates key header fields so the filename matches
        the timestamp inside the document.

    Args:
        feature_type (str): One of feature/refactor/epic/bug.
        target_dir (Path): Newly created active folder.
        feature_name (str): Slug/name inserted into doc placeholders.
        issue_field (str): Issue identifier string (e.g., #73).
        owner_field (str): Owner name.
        parent_field (str): Parent issue identifier or placeholder value.
        status_field (str): Status text to apply.
        version_field (str): Version token to apply.
        plan_timestamp (str): Timestamp token in `YYYY-MM-DDTHH-mm` (EST/ET).
        fs (FileSystem): File abstraction.

    Returns:
        Path | None: Materialized plan path, or None if no plan file exists.
    """

    template_plan = target_dir / PLAN_TIMESTAMP_TEMPLATE_NAME
    if fs.exists(template_plan):
        target_plan = target_dir / f"plan.{plan_timestamp}.md"

        # Rename first to avoid leaving template placeholders behind.
        fs.move(template_plan, target_plan)
        content = fs.read_text(target_plan)

        # Plan templates now expect full timestamp in the Last Updated field.
        updated_field = plan_timestamp

        content = set_header_placeholder(
            content,
            feature_name=feature_name,
            issue_field=issue_field,
            owner_field=owner_field,
            updated_field=updated_field,
            status_field=status_field,
            parent_field=parent_field,
            version_field=version_field,
        )
        fs.write_text(target_plan, content)
        return target_plan

    # Fallback for older templates.
    legacy = target_dir / "plan.md"
    if fs.exists(legacy):
        return legacy
    return None


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
    subprocess.run(  # noqa: S603
        [code_cmd, *[f.as_posix() for f in files]],
        check=False,
    )
    return True


def _apply_header_and_sections(
    path: Path,
    feature_name: str,
    issue_field: str,
    owner_field: str,
    updated_field: str,
    parent_field: str,
    status_field: str,
    version_field: str,
    fs: FileSystem,
    updates: list[tuple[str, str]],
) -> None:
    """Apply header metadata and optional section overrides to a doc file.

    Purpose:
        Keep spec and plan documents aligned with current template frontmatter
        while optionally seeding sections from potential docs.

    Args:
        path (Path): Document path to update.
        feature_name (str): Feature slug used in header placeholders.
        issue_field (str): Issue identifier string.
        owner_field (str): Owner name.
        updated_field (str): Last updated timestamp.
        parent_field (str): Parent issue identifier or placeholder value.
        status_field (str): Status text to apply.
        version_field (str): Version token to apply.
        fs (FileSystem): File abstraction layer.
        updates (list[tuple[str, str]]): Section name/body pairs to apply.
    """
    if not fs.exists(path):
        return
    content = fs.read_text(path)
    content = set_header_placeholder(
        content,
        feature_name,
        issue_field,
        owner_field,
        updated_field,
        status_field=status_field,
        parent_field=parent_field,
        version_field=version_field,
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
    parent_field: str,
    status_field: str,
    version_field: str,
    plan_updated_field: str,
    fs: FileSystem,
    sections: dict[str, str],
    plan_path: Path | None = None,
) -> list[Path]:
    """Populate active feature docs with header metadata and seeded content.

    Purpose:
        Apply consistent frontmatter metadata for newly created spec/plan files
        and inject any available context from potential documents.

    Args:
        feature_type (str): One of feature/refactor/epic/bug.
        feature_name (str): Feature slug used for placeholder replacement.
        target_dir (Path): Active feature folder.
        issue_field (str): Issue identifier string.
        owner_field (str): Owner name.
        updated_field (str): Timestamp for spec documents.
        parent_field (str): Parent issue identifier or placeholder value.
        status_field (str): Status text to apply.
        version_field (str): Version token to apply.
        plan_updated_field (str): Date or timestamp for plan documents.
        fs (FileSystem): File abstraction layer.
        sections (dict[str, str]): Seeded section content from potential docs.
        plan_path (Path | None): Optional plan document path.

    Returns:
        list[Path]: Files to open after creation.
    """
    files_to_open: list[Path] = []
    if feature_type == "feature":
        user_story = target_dir / "user-story.md"
        spec = target_dir / "spec.md"
        plan = plan_path or target_dir / "plan.md"
        _apply_header_and_sections(
            user_story,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            parent_field,
            status_field,
            version_field,
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
            parent_field,
            status_field,
            version_field,
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
            plan,
            feature_name,
            issue_field,
            owner_field,
            plan_updated_field,
            parent_field,
            status_field,
            version_field,
            fs,
            [],
        )
        files_to_open.extend([user_story, spec, plan])
    elif feature_type == "refactor":
        spec = target_dir / "spec.md"
        plan = plan_path or target_dir / "plan.md"
        _apply_header_and_sections(
            spec,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            parent_field,
            status_field,
            version_field,
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
            plan,
            feature_name,
            issue_field,
            owner_field,
            plan_updated_field,
            parent_field,
            status_field,
            version_field,
            fs,
            [],
        )
        files_to_open.extend([spec, plan])
    elif feature_type == "epic":
        initiative = target_dir / "initiative.md"
        _apply_header_and_sections(
            initiative,
            feature_name,
            issue_field,
            owner_field,
            updated_field,
            parent_field,
            status_field,
            version_field,
            fs,
            [],
        )
        files_to_open.append(initiative)
    elif feature_type == "bug":
        spec = target_dir / "spec.md"
        plan = plan_path or target_dir / "plan.md"

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
            parent_field,
            status_field,
            version_field,
            fs,
            updates,
        )
        # For bug plan docs, we want the full timestamp in the Date field.
        _apply_header_and_sections(
            plan,
            feature_name,
            issue_field,
            owner_field,
            plan_updated_field,
            parent_field,
            status_field,
            version_field,
            fs,
            [],
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
    now_provider: Callable[[], datetime] | None = None,
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
    issue_field = f"#{normalized_issue_number}" if normalized_issue_number else "TBD"
    if issue_meta:
        issue_field = f"#{issue_meta.number}"
    owner_field = issue_meta.author if issue_meta else "TBD"
    parent_field = "none"
    status_field = "Draft"
    version_field = "0.1"

    # One timestamp token is generated per folder creation and used consistently
    # for any timestamped plan file names and their document bodies.
    plan_timestamp = get_est_timestamp(now_provider)
    updated_field = plan_timestamp
    plan_path = materialize_plan_file(
        feature_type=feature_type,
        target_dir=target_dir,
        feature_name=feature_name,
        issue_field=issue_field,
        owner_field=owner_field,
        parent_field=parent_field,
        status_field=status_field,
        version_field=version_field,
        plan_timestamp=plan_timestamp,
        fs=filesystem,
    )

    # If we materialized a plan file, use its date (not the issue updated date)
    # when updating plan headers for non-bug types.
    plan_updated_field = plan_timestamp

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
        parent_field,
        status_field,
        version_field,
        plan_updated_field,
        filesystem,
        sections,
        plan_path=plan_path,
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

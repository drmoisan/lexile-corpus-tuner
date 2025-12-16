"""Promote a potential feature file to a GitHub issue using the gh CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

PLACEHOLDER = "(not provided in potential file)"
ISSUE_URL_PATTERN = re.compile(r"https?://\S+/issues/(\d+)")
PROMOTION_TYPES = ("epic", "feature", "refactor", "bug")


class PromotionError(Exception):
    """Raised when a promotion precondition fails."""


@dataclass
class GhResult:
    output: list[str]
    exit_code: int


class GhClient(Protocol):
    def is_authenticated(self) -> bool: ...

    def issue_create(self, title: str, body: str, promotion_type: str) -> GhResult: ...

    def issue_view(self, issue_number: str) -> GhResult: ...


@dataclass
class RealGhClient(GhClient):
    gh_path: str | None = None

    def __post_init__(self) -> None:
        if self.gh_path is None:
            self.gh_path = shutil.which("gh")
        if not self.gh_path:
            raise FileNotFoundError(
                "gh CLI not found on PATH. Install gh and authenticate first."
            )

    def is_authenticated(self) -> bool:
        """Check if gh CLI is authenticated by running gh auth status."""
        gh_exe = self.gh_path
        if gh_exe is None:
            return False

        result = subprocess.run(  # noqa: S603
            [gh_exe, "auth", "status"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _run(self, args: list[str], body: str | None = None) -> GhResult:
        gh_exe = self.gh_path
        if gh_exe is None:
            raise RuntimeError("gh CLI path was not resolved")

        proc: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
            [gh_exe, *args],
            input=body,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + stderr
        return GhResult(output=combined.splitlines(), exit_code=int(proc.returncode))

    def issue_create(self, title: str, body: str, promotion_type: str) -> GhResult:
        args = [
            "issue",
            "create",
            "--title",
            title,
            "--body-file",
            "-",
            "--label",
            promotion_type,
        ]
        return self._run(args, body)

    def issue_view(self, issue_number: str) -> GhResult:
        args = [
            "issue",
            "view",
            issue_number,
            "--json",
            "number,title,url,author,updatedAt",
        ]
        return self._run(args)


class FileSystem(Protocol):
    def resolve_path(self, path_str: str) -> Path: ...

    def exists(self, path: Path) -> bool: ...

    def read_text(self, path: Path) -> str: ...

    def write_text(self, path: Path, content: str) -> None: ...

    def write_lines(self, path: Path, lines: Iterable[str]) -> None: ...

    def ensure_dir(self, path: Path) -> None: ...

    def move(self, src: Path, dest: Path) -> None: ...


@dataclass
class RealFileSystem(FileSystem):
    def resolve_path(self, path_str: str) -> Path:
        return Path(path_str).expanduser().resolve()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def write_lines(self, path: Path, lines: Iterable[str]) -> None:
        joined = "\n".join(lines)
        path.write_text(joined, encoding="utf-8")

    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def move(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


@dataclass
class PromotionOutcome:
    exit_code: int
    messages: list[str]
    destination: Path | None = None


def _resolve_workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def get_feature_name(content: str, file_path: Path) -> str:
    heading_match = re.search(r"^\s*#\s+(.+)$", content, flags=re.MULTILINE)
    if heading_match:
        feature_name = heading_match.group(1).replace("(Potential)", "").strip()
        if feature_name:
            return feature_name

    name = file_path.name
    return name[:-3] if name.lower().endswith(".md") else name


def get_feature_path(feature_name: str) -> str:
    replaced = re.sub(r"\s+", "_", feature_name)
    return re.sub(r"[^A-Za-z0-9_-]", "", replaced)


def get_section(content: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = rf"^##\s+{escaped}\s*\r?\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def build_body(
    problem: str,
    behavior: str,
    criteria: str,
    constraints: str,
    tests: str,
    relative_path: str,
) -> str:
    return (
        f"## Problem / Why\n{problem}\n\n"
        f"## Proposed Behavior\n{behavior}\n\n"
        f"## Acceptance Criteria\n{criteria}\n\n"
        f"## Constraints & Risks\n{constraints}\n\n"
        f"## Test Conditions\n{tests}\n\n"
        f"## Source\nFrom: {relative_path}\n"
    )


def parse_issue_reference(output: Iterable[str]) -> tuple[str | None, str | None]:
    text = "\n".join(output)
    match = ISSUE_URL_PATTERN.search(text)
    if not match:
        return None, None
    return match.group(0), match.group(1)


def _extract_last_updated(issue_json: str) -> str | None:
    try:
        data = json.loads(issue_json)
    except json.JSONDecodeError:
        return None

    updated_raw = data.get("updatedAt")
    if not isinstance(updated_raw, str):
        return None

    try:
        dt = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.date().isoformat()


def _find_meta_end(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("## "):
            return idx
    return len(lines)


def _set_line_value(lines: list[str], label: str, value: str, meta_end: int) -> int:
    pattern = re.compile(rf"^- {re.escape(label)}:")
    for idx, line in enumerate(lines):
        if pattern.match(line):
            lines[idx] = f"- {label}: {value}"
            return meta_end
    lines.insert(meta_end, f"- {label}: {value}")
    return meta_end + 1


def update_metadata_lines(
    lines: list[str],
    feature_name: str,
    issue_number: str,
    issue_url: str,
    last_updated: str | None,
    feature_path: str,
) -> list[str]:
    if lines:
        lines[0] = f"# {feature_name} (Issue #{issue_number})"

    meta_end = _find_meta_end(lines)
    meta_end = _set_line_value(lines, "Issue", f"#{issue_number}", meta_end)
    meta_end = _set_line_value(lines, "Issue URL", issue_url, meta_end)
    if last_updated:
        meta_end = _set_line_value(lines, "Last Updated", last_updated, meta_end)
    status_value = (
        f"Promoted -> docs/features/active/{feature_path}/ (Issue #{issue_number})"
    )
    _set_line_value(lines, "Status", status_value, meta_end)
    return lines


def _default(message: str) -> None:
    print(message)


def promote_potential(
    potential_path: str,
    promotion_type: str = "feature",
    *,
    fs: FileSystem | None = None,
    gh: GhClient | None = None,
    workspace: Path | None = None,
    emit: Callable[[str], None] = _default,
) -> PromotionOutcome:
    if promotion_type not in PROMOTION_TYPES:
        raise PromotionError(f"Invalid promotion type: {promotion_type}")

    filesystem = fs or RealFileSystem()
    gh_client = gh or RealGhClient()
    workspace_path = workspace or _resolve_workspace()

    if not gh_client.is_authenticated():
        raise PromotionError(
            "GitHub CLI is not authenticated. Run 'gh auth login' first."
        )

    resolved = filesystem.resolve_path(potential_path)
    if not filesystem.exists(resolved):
        raise PromotionError(f"Potential file not found: {potential_path}")

    content = filesystem.read_text(resolved)
    if not content.strip():
        raise PromotionError(f"Potential file is empty: {resolved}")

    feature_name = get_feature_name(content, resolved)
    feature_path = get_feature_path(feature_name)
    issue_title = f"Feature: {feature_name}"

    problem = get_section(content, "Problem / Why") or PLACEHOLDER
    behavior = get_section(content, "Proposed Behavior") or PLACEHOLDER
    criteria = get_section(content, "Acceptance Criteria (early draft)") or PLACEHOLDER
    constraints = get_section(content, "Constraints & Risks") or PLACEHOLDER
    tests = get_section(content, "Test Conditions to Consider") or PLACEHOLDER

    try:
        relative_path = os.path.relpath(resolved, workspace_path)
    except ValueError:
        relative_path = str(resolved)

    body = build_body(problem, behavior, criteria, constraints, tests, relative_path)

    messages: list[str] = []

    def _emit(msg: str) -> None:
        messages.append(msg)
        emit(msg)

    _emit(f"Creating issue: {issue_title} (label: {promotion_type})")
    create_result = gh_client.issue_create(issue_title, body, promotion_type)

    if create_result.exit_code != 0:
        output_lines = create_result.output or [
            f"gh CLI exited with code {create_result.exit_code}"
        ]
        for line in output_lines:
            _emit(line)
        return PromotionOutcome(exit_code=create_result.exit_code, messages=messages)

    for line in create_result.output:
        _emit(line)

    issue_url, issue_number = parse_issue_reference(create_result.output)

    last_updated: str | None = None
    if issue_number:
        view_result = gh_client.issue_view(issue_number)
        if view_result.exit_code == 0 and view_result.output:
            last_updated = _extract_last_updated("\n".join(view_result.output))

    if issue_number and issue_url:
        lines = content.splitlines()
        updated_lines = update_metadata_lines(
            lines,
            feature_name,
            issue_number,
            issue_url,
            last_updated,
            feature_path,
        )
        filesystem.write_lines(resolved, updated_lines)
        _emit(f"Updated potential file with issue metadata: {resolved}")

    promoted_dir = workspace_path / "docs" / "features" / "potential" / "promoted"
    filesystem.ensure_dir(promoted_dir)
    dest_path = promoted_dir / resolved.name
    filesystem.move(resolved, dest_path)
    _emit(f"Moved potential file to promoted folder: {dest_path}")

    return PromotionOutcome(exit_code=0, messages=messages, destination=dest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GitHub issue from a potential feature file using gh CLI.",
    )
    parser.add_argument(
        "--potential-path",
        required=True,
        help="Path to the potential feature markdown file.",
    )
    parser.add_argument(
        "--promotion-type",
        choices=PROMOTION_TYPES,
        default="feature",
        help="Promotion type label to attach to the issue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outcome = promote_potential(
            potential_path=args.potential_path,
            promotion_type=args.promotion_type,
        )
    except (PromotionError, FileNotFoundError) as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    except subprocess.SubprocessError as exc:  # pragma: no cover
        print(f"gh CLI execution failed: {exc}")
        raise SystemExit(1) from exc

    raise SystemExit(outcome.exit_code)


if __name__ == "__main__":
    main()

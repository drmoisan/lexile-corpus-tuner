"""Helper routines for PR context rendering and summarization."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import (
    IssueDetails,
    PullRequestDetails,
    format_list,
    section,
    truncate_lines,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .git import GitClient

__all__ = [
    "append_generation_timestamp",
    "bucket_text",
    "extract_digest_bullets",
    "is_scoping_doc",
    "issue_appendix",
    "issue_digest",
    "last_with_truncation",
    "parse_name_status_map",
    "parse_numstat_detailed",
    "parse_section",
    "pr_appendix",
    "pr_digest",
    "scoping_doc_changes",
]


def last_with_truncation(items: list[str], limit: int) -> tuple[list[str], bool]:
    if len(items) <= limit:
        return items, False
    return items[-limit:], True


def extract_digest_bullets(body: str, *, headings: list[str], limit: int) -> list[str]:
    bullets: list[str] = []
    for heading in headings:
        section_text = parse_section(body, heading)
        if not section_text:
            continue
        for line in section_text.splitlines():
            if not line.strip():
                continue
            cleaned = line.lstrip("-*").strip()
            bullets.append(f"{heading}: {cleaned}")
            if len(bullets) >= limit:
                return bullets
    return bullets[:limit]


def issue_digest(issue: IssueDetails) -> str:
    bullets = extract_digest_bullets(
        issue.body,
        headings=[
            "Why",
            "Context",
            "Root Cause",
            "Constraints",
            "Acceptance Criteria",
            "Test Strategy",
            "Risks",
            "Verification",
            "Follow-ups",
        ],
        limit=8,
    )
    if not bullets:
        bullets.append(f"State: {issue.state}")
        if issue.labels:
            bullets.append(f"Labels: {', '.join(issue.labels)}")

    selected_comments, truncated = last_with_truncation(issue.comments, 3)
    comment_block = (
        "\n".join(f"- {comment}" for comment in selected_comments)
        if selected_comments
        else "(no comments)"
    )
    if truncated:
        comment_block += "\nTRUNCATED: last 3 comments shown"

    metadata = [
        f"Identifier: {issue.number}",
        f"Title: {issue.title}",
        f"Author: {issue.author}",
        f"Assignees: {', '.join(issue.assignees) if issue.assignees else '(none)'}",
        f"Labels: {', '.join(issue.labels) if issue.labels else '(none)'}",
        f"State: {issue.state}",
        f"Last updated: {issue.updated_at}",
    ]
    return "\n".join(
        [
            "\n".join(metadata),
            "Key bullets:",
            "\n".join(f"- {entry}" for entry in bullets),
            "",
            "Recent comments:",
            comment_block,
        ]
    )


def pr_digest(pr: PullRequestDetails) -> str:
    bullets = extract_digest_bullets(
        pr.body,
        headings=[
            "Why",
            "Context",
            "Root Cause",
            "Constraints",
            "Acceptance Criteria",
            "Test Strategy",
            "Risks",
            "Verification",
            "Follow-ups",
        ],
        limit=8,
    )
    if not bullets:
        if pr.files_changed:
            bullets.append(
                f"Touches files: {', '.join(pr.files_changed[:3])}"
                + (" ..." if len(pr.files_changed) > 3 else "")
            )
        bullets.append(f"State: {pr.state}")

    metadata = [
        f"Identifier: {pr.number}",
        f"Title: {pr.title}",
        f"Author: {pr.author}",
        f"Base/Head: {pr.base_ref} <- {pr.head_ref}",
        f"Last updated: {pr.updated_at}",
    ]
    return "\n".join(
        [
            "\n".join(metadata),
            "Key bullets:",
            "\n".join(f"- {entry}" for entry in bullets),
        ]
    )


def issue_appendix(issue: IssueDetails) -> str:
    body_text = truncate_lines(issue.body, 120)
    comments, truncated = last_with_truncation(issue.comments, 10)
    comment_text = (
        "\n".join(f"- {c}" for c in comments) if comments else "(no comments)"
    )
    if truncated:
        comment_text += "\nTRUNCATED: last 10 comments shown"
    user_story_block = ""
    if issue.user_story_content:
        user_story_block = "\n".join(
            [
                "",
                f"User story ({issue.user_story_path or 'user-story.md'}):",
                truncate_lines(issue.user_story_content, 120),
            ]
        )
    return "\n".join(
        [
            section(f"Issue {issue.number}: {issue.title}"),
            f"State: {issue.state}",
            f"Labels: {', '.join(issue.labels) if issue.labels else '(none)'}",
            f"Assignees: {', '.join(issue.assignees) if issue.assignees else '(none)'}",
            f"Author: {issue.author}",
            f"Created: {issue.created_at}",
            f"Updated: {issue.updated_at}",
            "",
            body_text,
            "",
            "Comments:",
            comment_text,
            user_story_block,
        ]
    )


def pr_appendix(pr: PullRequestDetails) -> str:
    body_text = truncate_lines(pr.body, 120)
    return "\n".join(
        [
            section(f"Pull Request {pr.number}: {pr.title}"),
            f"State: {pr.state}",
            f"Author: {pr.author}",
            f"Base: {pr.base_ref}",
            f"Head: {pr.head_ref}",
            f"Created: {pr.created_at}",
            f"Updated: {pr.updated_at}",
            f"Merged: {pr.merged_at or '(not merged)'}",
            f"Labels: {', '.join(pr.labels) if pr.labels else '(none)'}",
            f"Assignees: {', '.join(pr.assignees) if pr.assignees else '(none)'}",
            "",
            body_text,
            "",
            "Auto-close issues (from this PR):",
            format_list(pr.closing_issues, "(none)"),
            "",
            "Files (first 25):",
            format_list(pr.files_changed[:25], "(none)"),
        ]
    )


def parse_numstat_detailed(
    numstat_text: str,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    adds_total = 0
    dels_total = 0
    per_file: dict[str, tuple[int, int]] = {}
    for raw_line in numstat_text.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        if len(parts) < 3:
            continue
        add_part, del_part, file_part = parts[0], parts[1], parts[2]
        add_count = int(add_part) if add_part.isdigit() else 0
        del_count = int(del_part) if del_part.isdigit() else 0
        adds_total += add_count
        dels_total += del_count
        per_file[format_diff_path(file_part)] = (add_count, del_count)
    return adds_total, dels_total, per_file


def parse_name_status_map(name_status_text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in name_status_text.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        path = format_diff_path(parts[-1].strip())
        mapping[path] = status
    return mapping


def is_scoping_doc(path: str) -> bool:
    lowered = path.lower()
    return bool(
        lowered.startswith("docs/features/")
        and (
            lowered.endswith("/spec.md")
            or lowered.endswith("/plan.md")
            or lowered.endswith("/bug-remediation-plan.md")
            or lowered.endswith("/user-story.md")
            or lowered.endswith("/readme.md")
        )
    )


def scoping_doc_changes(
    *,
    git: GitClient,
    merge_base: str | None,
    head_sha: str | None,
    root: Path,
    name_status_text: str,
    numstat_details: dict[str, tuple[int, int]],
) -> list[tuple[str, bool, list[str], str | None]]:
    if not merge_base or not head_sha:
        return []
    changes: list[tuple[str, bool, list[str], str | None]] = []
    name_status_map = parse_name_status_map(name_status_text)
    for path, status in name_status_map.items():
        if not is_scoping_doc(path):
            continue
        additions, deletions = numstat_details.get(path, (0, 0))
        reasons: list[str] = []
        material = False
        if status.startswith("A"):
            material = True
            reasons.append("new scoping doc")
        if additions + deletions >= 15:
            material = True
            reasons.append(">=15 lines changed")

        diff_text = git.diff_range(["--unified=0", merge_base, head_sha, "--", path])
        heading_touched = False
        for line in diff_text.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            stripped = line.lstrip("+").strip()
            if any(
                stripped.lower().startswith(prefix.lower())
                for prefix in (
                    "## Context",
                    "## Root Cause",
                    "## Proposed Fix",
                    "## Acceptance Criteria",
                    "## Test Strategy",
                    "## Risks",
                )
            ):
                heading_touched = True
                break
        if heading_touched:
            material = True
            reasons.append("key section touched")

        added_lines = [
            line.lstrip("+").strip()
            for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        if added_lines and all(
            (not line or line.startswith("[") or line.startswith("http"))
            for line in added_lines
        ):
            reasons.append("link/whitespace-only changes")
            if (
                not heading_touched
                and additions + deletions < 15
                and not status.startswith("A")
            ):
                material = False

        excerpt = None
        doc_path = root / path
        if material and doc_path.exists():
            content = doc_path.read_text(encoding="utf-8")
            excerpt_parts: list[str] = []
            for heading in (
                "Acceptance Criteria",
                "Root Cause",
                "Proposed Fix",
                "Test Strategy",
            ):
                section_text = parse_section(content, heading)
                if section_text:
                    excerpt_parts.append(
                        f"{heading}:\n{truncate_lines(section_text, 40)}"
                    )
            excerpt = "\n\n".join(excerpt_parts[:3]) if excerpt_parts else None

        changes.append((path, material, reasons, excerpt))
    return changes


def bucket_text(name: str, entries: list[tuple[str, tuple[int, int]]]) -> str:
    if not entries:
        return f"{name}: 0 files"
    sorted_entries = sorted(
        entries, key=lambda item: item[1][0] + item[1][1], reverse=True
    )
    lines = [
        f"{name}: {len(entries)} files",
        *(f"- {path} (+{adds}/-{dels})" for path, (adds, dels) in sorted_entries[:10]),
    ]
    return "\n".join(lines)


def parse_section(markdown: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = rf"^##\s+{escaped}\s*\r?\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def format_diff_path(path_text: str | None) -> str:
    from .render import format_diff_path as _fmt

    return _fmt(path_text) if path_text is not None else ""


def append_generation_timestamp() -> str:
    """Generate a timestamp section showing when context was collected.

    Returns:
        Formatted timestamp section with UTC time
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    return section("Context generated") + "\n" + timestamp + "\n"

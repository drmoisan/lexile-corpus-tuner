"""Shared models and helpers for PR context collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

SECTION_LINE = "===== {title} ====="
CONVENTIONAL_TYPES = (
    "feat",
    "fix",
    "refactor",
    "perf",
    "docs",
    "test",
    "chore",
    "build",
    "ci",
    "style",
)


@dataclass
class CommandResult:
    """Represents the outcome of a shell command."""

    stdout: str
    stderr: str
    code: int


@dataclass
class IssueDetails:
    """Issue metadata, body, comments, and optional user story content."""

    number: str
    title: str
    body: str
    comments: list[str]
    user_story_path: str | None = None
    user_story_content: str | None = None


@dataclass
class PullRequestDetails:
    """Pull request metadata and body content."""

    number: str
    title: str
    body: str
    closing_issues: list[str]


@dataclass
class FeatureDocExcerpt:
    """Feature documentation excerpts and referenced issues."""

    feature: str
    excerpt: str
    issue_refs: list[str]


@dataclass
class PRContextResult:
    """Structured result from building the PR comparison section."""

    text: str
    referenced_issues: list[str]
    referenced_prs: list[str]
    verified_closing: list[str]


def section(title: str) -> str:
    return "\n" + SECTION_LINE.format(title=title) + "\n"


def truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_reference(ref: str) -> str:
    return ref.lstrip("#").strip()


def find_user_story_link(body: str) -> str | None:
    if not body:
        return None

    match = re.search(r"\(([^)]+user-story\.md)\)", body, flags=re.IGNORECASE)
    candidate = match.group(1) if match else None

    if not candidate:
        fallback = re.search(r"\b\S*user-story\.md\b", body, flags=re.IGNORECASE)
        candidate = fallback.group(0) if fallback else None

    if not candidate:
        return None

    github_blob = re.search(
        r"github\.com/[^/]+/[^/]+/blob/[^/]+/(.+user-story\.md)",
        candidate,
        flags=re.IGNORECASE,
    )
    if github_blob:
        return github_blob.group(1)

    return candidate.lstrip("/")


def format_list(values: Iterable[str], empty_text: str) -> str:
    values_list = [value for value in values if value]
    if not values_list:
        return empty_text
    return "\n".join(f"- {item}" for item in values_list)

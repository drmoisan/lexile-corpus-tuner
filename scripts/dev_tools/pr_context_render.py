"""Rendering and extraction helpers for PR context collection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pr_context_git import GitClient
from .pr_context_models import (
    CONVENTIONAL_TYPES,
    FeatureDocExcerpt,
    IssueDetails,
    PRContextResult,
    PullRequestDetails,
    format_list,
    normalize_reference,
    section,
    truncate,
)


class GhLike(Protocol):
    def ensure_available(self) -> None: ...

    def classify_entity(self, number: str) -> str | None: ...


def select_default_base(git: GitClient) -> str | None:
    candidates = [
        "origin/main",
        "origin/master",
        "main",
        "master",
        "origin/develop",
        "develop",
    ]
    for ref in candidates:
        result = git.run(["rev-parse", "--verify", "--quiet", ref], allow_error=True)
        if result.code == 0 and result.stdout.strip():
            return ref
    return None


def format_diff_path(path_text: str | None) -> str:
    if path_text is None:
        return ""
    if path_text.strip() == "":
        return path_text

    trimmed = path_text.strip().strip('"')
    trimmed = re.sub(r"\{[^{}]*\s=>\s([^{}]*)\}", r"\1", trimmed)

    arrow_match = re.match(r"^\s*(.+?)\s=>\s(.+?)\s*$", trimmed)
    if arrow_match:
        return arrow_match.group(2)
    return trimmed


def convert_numstat(numstat_text: str) -> tuple[int, int, list[str]]:
    adds = 0
    dels = 0
    files: list[str] = []

    for raw_line in numstat_text.splitlines():
        if not raw_line.strip():
            continue

        parts = raw_line.split("\t")
        if len(parts) < 3:
            continue

        add_part, del_part, file_part = parts[0], parts[1], parts[2]
        if add_part.isdigit():
            adds += int(add_part)
        if del_part.isdigit():
            dels += int(del_part)
        files.append(file_part)

    return adds, dels, files


def extension_summary(files: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for raw in files:
        name = format_diff_path(raw)
        ext = "(unknown)"
        try:
            suffix = Path(name).suffix
            ext = suffix if suffix else "(noext)"
        except ValueError:
            fallback = re.search(r"\.([A-Za-z0-9_]+)$", name)
            ext = f".{fallback.group(1)}" if fallback else "(unknown)"

        counts[ext] = counts.get(ext, 0) + 1

    lines = [f"{counts[k]:8d}  {k}" for k in sorted(counts)]
    return "\n".join(lines)


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


def extract_merge_pr_numbers(subjects: Iterable[str]) -> list[str]:
    numbers: set[str] = set()
    pattern = re.compile(r"Merge pull request #(\d+)", re.IGNORECASE)
    for subj in subjects:
        match = pattern.search(subj)
        if match:
            numbers.add(f"#{match.group(1)}")
    return sorted(numbers)


def summarize_conventional_commits(subjects: str) -> str:
    counts = {key: 0 for key in CONVENTIONAL_TYPES}
    counts["other"] = 0

    for line in subjects.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(feat|fix|refactor|perf|docs|test|chore|build|ci|style)(\(|!|:)",
            line,
        )
        label = match.group(1) if match else "other"
        counts[label] += 1

    non_zero = [(k, v) for k, v in counts.items() if v > 0]
    if not non_zero:
        return "(no recognizable conventional commit types)"
    return "\n".join(f"{name:<9} : {value}" for name, value in non_zero)


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
    for feature in sorted(features):
        spec_path = root / "docs" / "features" / "active" / feature / "spec.md"
        plan_path = root / "docs" / "features" / "active" / feature / "plan.md"

        spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
        plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""

        spec_parts: list[str] = []
        for heading in (
            "Context",
            "Root Cause Analysis",
            "Proposed Fix",
            "Acceptance Criteria",
        ):
            section_text = parse_section(spec_text, heading)
            if section_text:
                spec_parts.append(f"{heading}: {truncate(section_text)}")

        plan_tasks = completed_plan_tasks(plan_text)
        plan_section = (
            "\n".join(f"- {task}" for task in plan_tasks) if plan_tasks else ""
        )

        lines: list[str] = [section(f"Feature doc: {feature}")]
        if spec_parts:
            lines.append("Spec excerpts:\n" + "\n\n".join(spec_parts))
        if plan_section:
            lines.append("Plan completed tasks:\n" + plan_section)
        if len(lines) == 1:
            lines.append("(no spec/plan excerpts found)")

        excerpt_text = "\n".join(lines)
        issue_refs = extract_issue_references(spec_text + "\n" + plan_text)
        excerpts.append(
            FeatureDocExcerpt(
                feature=feature, excerpt=excerpt_text, issue_refs=issue_refs
            )
        )

    return excerpts


def format_issue_details(issue: IssueDetails) -> str:
    comments_text = format_list(issue.comments, "(no comments)")
    lines = [
        section(f"Issue {issue.number}: {issue.title}"),
        truncate(issue.body, 1200),
        "",
        "Comments:",
        comments_text,
    ]
    if issue.user_story_content:
        lines.extend(
            [
                "",
                f"User story ({issue.user_story_path or 'user-story.md'}):",
                truncate(issue.user_story_content, 1200),
            ]
        )
    return "\n".join(lines)


def format_pr_details(pr: PullRequestDetails) -> str:
    return "\n".join(
        [
            section(f"Pull Request {pr.number}: {pr.title}"),
            truncate(pr.body, 1200),
            "",
            "Auto-close issues (from this PR):",
            format_list(pr.closing_issues, "(none)"),
        ]
    )


def build_close_candidates_section(
    *,
    verified: list[str],
    author_asserted: list[str],
    referenced: list[str],
) -> str:
    return "\n".join(
        [
            section("Close candidates"),
            "Auto-close issues (verified from GitHub PR metadata):",
            format_list(verified, "(no PR detected or no verified closing issues)"),
            "",
            "Auto-close issues (author asserted):",
            format_list(author_asserted, "(none)"),
            "",
            "Referenced issues (detected):",
            format_list(referenced, "(none)"),
        ]
    )


def build_pr_context(
    *,
    git: GitClient,
    gh: GhLike,
    base_ref: str | None,
    head_ref: str | None,
    include_untracked: bool,
    feature_issue_refs: Iterable[str] | None = None,
    current_pr: PullRequestDetails | None = None,
) -> PRContextResult:
    gh.ensure_available()
    branch_name = git.branch_name()
    upstream = git.upstream() or "(none)"

    remotes = git.remote_verbose()
    status_short = git.status_short()
    untracked = git.untracked() if include_untracked else ""
    untracked_display = untracked if untracked.strip() else "(none)"

    feature_issue_list = list(feature_issue_refs or [])
    referenced_issues: list[str] = []
    referenced_prs: list[str] = []
    verified_closing = current_pr.closing_issues if current_pr else []

    pr_block = ""
    try:
        resolved_base = base_ref or select_default_base(git)
        if not resolved_base:
            raise RuntimeError("Failed to resolve base ref (tried common defaults)")

        base = git.rev_parse(resolved_base)
        head = git.rev_parse(head_ref or "HEAD")
        merge_base = git.merge_base(base, head)
        rev_range = f"{merge_base}..{head}"

        oneline = git.log("--pretty=format:%h %ad %an %s", rev_range)
        subjects = git.log("--pretty=%s", rev_range)
        authors = git.log("--format=%an <%ae>", rev_range)
        authors_list = sorted(
            {line.strip() for line in authors.splitlines() if line.strip()}
        )

        name_status = git.diff_range(["--name-status", merge_base, head])
        numstat = git.diff_range(["--numstat", merge_base, head])
        shortstat = git.diff_range(["--shortstat", merge_base, head])
        stat = git.diff_range(["--stat", merge_base, head])

        additions, deletions, files = convert_numstat(numstat)
        ext_summary = extension_summary(files)
        merge_prs = extract_merge_pr_numbers(oneline.splitlines())

        issue_candidates = [
            ref
            for ref in extract_issue_references(oneline + "\n" + subjects)
            if ref not in merge_prs
        ]
        issues: list[str] = []
        prs: list[str] = []
        for ref in issue_candidates:
            number = normalize_reference(ref)
            entity = gh.classify_entity(number)
            if entity == "issue":
                issues.append(ref if ref.startswith("#") else f"#{ref}")
            elif entity == "pull":
                prs.append(ref if ref.startswith("#") else f"#{ref}")
            else:
                issues.append(ref)

        referenced_issues = sorted(set(issues))
        referenced_prs = sorted(set(prs + merge_prs))

        issues_display = ", ".join(sorted(set(referenced_issues + feature_issue_list)))
        if not issues_display:
            issues_display = "(none)"
        prs_display = ", ".join(referenced_prs) if referenced_prs else "(none)"

        oneline_display = oneline if oneline.strip() else "(none)"
        authors_display = "\n".join(authors_list) if authors_list else "(none)"
        name_status_display = name_status if name_status.strip() else "(none)"
        short_display = shortstat if shortstat.strip() else "(none)"
        ext_display = ext_summary if ext_summary else "(none)"
        stat_display = stat if stat.strip() else "(none)"

        pr_block = "\n".join(
            [
                section("PR Comparison"),
                f"Base: {base_ref or resolved_base}",
                f"Head: {head_ref or branch_name}",
                f"Merge-base: {merge_base}",
                f"Range: {rev_range}\n",
                section("Commits in range"),
                oneline_display,
                "",
                section("Conventional commit type summary"),
                summarize_conventional_commits(subjects),
                "",
                section("Authors"),
                authors_display,
                "",
                section("Changed files (name-status)"),
                name_status_display,
                "",
                section("Diff shortstat"),
                short_display,
                "",
                section("Additions/Deletions totals (from numstat)"),
                f"Additions: {additions}\nDeletions: {deletions}\n",
                section("Files by extension"),
                ext_display,
                "",
                section("Referenced issues (detected)"),
                issues_display,
                "",
                section("PRs in range"),
                prs_display,
                "",
                section("Diff stat"),
                stat_display,
            ]
        )
    except Exception as exc:  # noqa: BLE001
        pr_block = section("PR Comparison") + f"(FAILED to compute PR context: {exc})\n"
        referenced_issues = []
        referenced_prs = []
        verified_closing = []

    intent = "\n".join(
        [
            section("PR Intent (edit before generating PR body)"),
            "Primary outcome:",
            "Impact (user/developer):",
            "Risks:",
            "Author-asserted autoclose issues:",
        ]
    )

    combined_text = "\n".join(
        [
            intent,
            section("Repository remotes"),
            remotes,
            "",
            section("Current branch"),
            branch_name,
            "",
            section("Upstream"),
            upstream,
            "",
            section("Status (short)"),
            status_short,
            "",
            section("Untracked files"),
            untracked_display,
            "",
            section("Working tree diff (staged)"),
            git.diff_name_status(staged=True),
            git.diff_patch(staged=True),
            "",
            section("Working tree diff (unstaged)"),
            git.diff_name_status(staged=False),
            git.diff_patch(staged=False),
            pr_block,
        ]
    )

    return PRContextResult(
        text=combined_text,
        referenced_issues=referenced_issues,
        referenced_prs=referenced_prs,
        verified_closing=sorted(set(verified_closing)),
    )


def extract_changed_paths(context_text: str) -> list[str]:
    paths: list[str] = []
    capture = False
    for line in context_text.splitlines():
        if line.startswith("===== Changed files"):
            capture = True
            continue
        if capture:
            if line.startswith("====="):
                break
            if line.strip() and "\t" in line:
                path_part = line.split("\t")[-1]
                paths.append(format_diff_path(path_part.strip()))
            elif line.strip():
                paths.append(format_diff_path(line.strip()))
    return paths

"""Rendering and extraction helpers for PR context collection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .git import GitClient
from .models import (
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

    @property
    def available(self) -> bool: ...


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


def resolve_feature_dir(base_dir: Path, feature: str) -> Path | None:
    """Resolve feature directory by exact match or fuzzy match."""
    direct = base_dir / feature
    if direct.exists():
        return direct

    if not base_dir.exists():
        return None

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


def read_text_file(path: Path) -> str:
    """Read text from path if it exists, otherwise return empty string."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def extract_features_from_paths(changed_files: Iterable[str]) -> set[str]:
    """Extract feature names from docs/features/active/** paths."""
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
    return features


def extract_spec_parts(spec_text: str) -> list[str]:
    """Extract relevant sections from spec document."""
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
    return spec_parts


def extract_plan_sections(plan_text: str) -> tuple[str, str]:
    """Extract completed tasks and verification block from plan."""
    plan_tasks = completed_plan_tasks(plan_text)
    plan_section = "\n".join(f"- {task}" for task in plan_tasks) if plan_tasks else ""
    test_plan_section = parse_section(plan_text, "Test Plan")
    verification_block = (
        "Plan verification notes:\n" + truncate(test_plan_section)
        if test_plan_section
        else ""
    )
    return plan_section, verification_block


def extract_story_parts(user_story_text: str, promoted_story_text: str) -> list[str]:
    """Extract story statement and problem sections from user story."""
    story_parts: list[str] = []
    story_statements = parse_section(user_story_text, "Story Statement")
    if story_statements:
        story_lines = [
            line.strip("- ") for line in story_statements.splitlines() if line.strip()
        ]
        if story_lines:
            story_parts.append(
                "Story Statement:\n" + "\n".join(f"- {line}" for line in story_lines)
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
    return story_parts


def build_excerpt_text(
    feature: str,
    story_parts: list[str],
    spec_parts: list[str],
    plan_section: str,
    verification_block: str,
) -> str:
    """Build the formatted excerpt text from collected parts."""
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
    return "\n".join(lines)


def gather_feature_excerpts(
    root: Path, changed_files: Iterable[str]
) -> list[FeatureDocExcerpt]:
    """Extract feature documentation excerpts from changed files."""
    features = extract_features_from_paths(changed_files)
    excerpts: list[FeatureDocExcerpt] = []
    base_dir = root / "docs" / "features" / "active"
    promoted_dir = root / "docs" / "features" / "potential" / "promoted"

    for feature in sorted(features):
        feature_dir = resolve_feature_dir(base_dir, feature)
        promoted_feature_dir = resolve_feature_dir(promoted_dir, feature)
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
            read_text_file(promoted_story_path) if promoted_story_path else ""
        )
        if promoted_story_path is not None and not user_story_path.exists():
            user_story_path = promoted_story_path

        user_story_text = read_text_file(user_story_path)
        if (
            not user_story_text
            and promoted_story_text
            and promoted_story_path is not None
        ):
            user_story_text = promoted_story_text
            user_story_path = promoted_story_path

        spec_text = read_text_file(spec_path)
        plan_text = read_text_file(plan_path)

        spec_parts = extract_spec_parts(spec_text)
        plan_section, verification_block = extract_plan_sections(plan_text)
        story_parts = extract_story_parts(user_story_text, promoted_story_text)
        excerpt_text = build_excerpt_text(
            feature, story_parts, spec_parts, plan_section, verification_block
        )

        context_files = [
            str(path.relative_to(root))
            for path in (spec_path, plan_path, user_story_path)
            if path and path.exists()
        ]
        issue_refs = extract_issue_references(
            "\n".join([spec_text, plan_text, user_story_text])
        )
        excerpts.append(
            FeatureDocExcerpt(
                feature=feature,
                excerpt=excerpt_text,
                issue_refs=issue_refs,
                context_files=context_files,
            )
        )

    return excerpts


def format_issue_details(issue: IssueDetails) -> str:
    comments_text = format_list(issue.comments, "(no comments)")
    lines = [
        section(f"Issue {issue.number}: {issue.title}"),
        f"State: {issue.state}",
        f"Author: {issue.author}",
        f"Labels: {', '.join(issue.labels) if issue.labels else '(none)'}",
        f"Assignees: {', '.join(issue.assignees) if issue.assignees else '(none)'}",
        f"Created: {issue.created_at}",
        f"Updated: {issue.updated_at}",
        "",
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
            f"State: {pr.state}",
            f"Author: {pr.author}",
            f"Base: {pr.base_ref}",
            f"Head: {pr.head_ref}",
            f"Created: {pr.created_at}",
            f"Updated: {pr.updated_at}",
            f"Merged: {pr.merged_at or '(not merged)'}",
            f"Labels: {', '.join(pr.labels) if pr.labels else '(none)'}",
            f"Assignees: {', '.join(pr.assignees) if pr.assignees else '(none)'}",
            truncate(pr.body, 1200),
            "",
            "Auto-close issues (from this PR):",
            format_list(pr.closing_issues, "(none)"),
            "",
            "Files (first 15):",
            format_list(pr.files_changed[:15], "(none)"),
        ]
    )


def build_close_candidates_section(
    *,
    verified: list[str],
    author_asserted: list[str],
    referenced: list[str],
    verified_reason: str,
    author_reason: str,
) -> str:
    author_auto_close = sorted(set(author_asserted + referenced))
    return "\n".join(
        [
            section("Close candidates"),
            "Auto-close issues (verified from GitHub PR metadata):",
            format_list(verified, verified_reason),
            "",
            "Auto-close issues (author asserted):",
            format_list(author_auto_close, author_reason),
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
    gh_available: bool | None = None,
) -> PRContextResult:
    gh_available = (
        gh.available
        if gh_available is None and hasattr(gh, "available")
        else gh_available
    )
    gh_available = True if gh_available is None else gh_available
    if gh_available:
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
    verified_closing = (
        current_pr.closing_issues if (current_pr and gh_available) else []
    )
    invalid_references: list[str] = []

    resolved_base: str | None = None
    base_sha: str | None = None
    head_sha: str | None = None
    head_ref_resolved: str | None = head_ref or branch_name
    merge_base: str | None = None
    rev_range: str | None = None

    pr_block = ""
    base_warning: str | None = None
    try:
        requested_base = base_ref
        resolved_base = base_ref or select_default_base(git)
        if not resolved_base:
            raise RuntimeError("Failed to resolve base ref (tried common defaults)")

        if not resolved_base.startswith("origin/"):
            remote_candidate = f"origin/{resolved_base}"
            remote_probe = git.run(
                ["rev-parse", "--verify", "--quiet", remote_candidate],
                allow_error=True,
            )
            if remote_probe.code == 0 and remote_probe.stdout.strip():
                resolved_base = remote_candidate
            elif requested_base:
                base_warning = (
                    "WARNING: Requested base is local and may be stale; prefer "
                    f"origin/{requested_base}"
                )

        base_sha = git.rev_parse(resolved_base)
        head_ref_resolved = head_ref or branch_name
        head_sha = git.rev_parse(head_ref_resolved or "HEAD")
        merge_base = git.merge_base(base_sha, head_sha)
        rev_range = f"{merge_base}..{head_sha}"

        oneline = git.log("--pretty=format:%h %ad %an %s", rev_range)
        subjects = git.log("--pretty=%s", rev_range)
        authors = git.log("--format=%an <%ae>", rev_range)
        authors_list = sorted(
            {line.strip() for line in authors.splitlines() if line.strip()}
        )

        name_status = git.diff_range(["--name-status", merge_base, head_sha])
        numstat = git.diff_range(["--numstat", merge_base, head_sha])
        shortstat = git.diff_range(["--shortstat", merge_base, head_sha])
        stat = git.diff_range(["--stat", merge_base, head_sha])

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
            if gh_available:
                entity = gh.classify_entity(number)
            else:
                entity = None
            formatted_ref = ref if ref.startswith("#") else f"#{ref}"
            if entity == "issue":
                issues.append(formatted_ref)
            elif entity == "pull":
                prs.append(formatted_ref)
            else:
                if gh_available:
                    invalid_references.append(formatted_ref)
                else:
                    issues.append(formatted_ref)
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

        block_lines = [
            section("PR Comparison"),
            f"Base ref (requested): {requested_base or '(default)'}",
            f"Base ref (resolved): {resolved_base} @ {base_sha}",
            f"Head ref (resolved): {head_ref_resolved} @ {head_sha}",
            f"Merge-base: {merge_base}",
        ]
        if base_warning:
            block_lines.append(f"Base warning: {base_warning}")
        block_lines.extend(
            [
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
        pr_block = "\n".join(block_lines)
    except Exception as exc:  # noqa: BLE001
        pr_block = section("PR Comparison") + f"(FAILED to compute PR context: {exc})\n"
        referenced_issues = []
        referenced_prs = []
        verified_closing = []
        invalid_references = []
        resolved_base = None
        base_sha = None
        head_sha = None
        head_ref_resolved = head_ref or branch_name
        merge_base = None
        rev_range = None

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
        invalid_references=sorted(set(invalid_references)),
        base_ref=base_ref,
        resolved_base=resolved_base,
        base_sha=base_sha,
        head_ref=head_ref_resolved,
        head_sha=head_sha,
        merge_base=merge_base,
        rev_range=rev_range,
        gh_available=gh_available,
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

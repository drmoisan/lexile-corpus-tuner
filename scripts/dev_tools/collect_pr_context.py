"""Collect Git repository context for pull request authorship."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .pr_context_gh import GhClient
from .pr_context_git import CommandRunner, GitClient, SubprocessRunner
from .pr_context_models import (
    CommandResult,
    FeatureDocExcerpt,
    IssueDetails,
    PRContextResult,
    PullRequestDetails,
    find_user_story_link,
    format_list,
    section,
    truncate_lines,
)
from .pr_context_render import (
    build_close_candidates_section,
    build_pr_context,
    completed_plan_tasks,
    convert_numstat,
    extension_summary,
    extract_changed_paths,
    extract_issue_references,
    extract_merge_pr_numbers,
    format_diff_path,
    format_issue_details,
    format_pr_details,
    gather_feature_excerpts,
    parse_section,
    select_default_base,
)  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CommandResult",
    "FeatureDocExcerpt",
    "IssueDetails",
    "PRContextResult",
    "PullRequestDetails",
    "GitClient",
    "SubprocessRunner",
    "CommandRunner",
    "GhClient",
    "build_pr_context",
    "build_close_candidates_section",
    "completed_plan_tasks",
    "convert_numstat",
    "extension_summary",
    "extract_issue_references",
    "extract_merge_pr_numbers",
    "format_diff_path",
    "format_issue_details",
    "format_pr_details",
    "gather_feature_excerpts",
    "select_default_base",
    "find_user_story_link",
    "section",
    "extract_changed_paths",
    "collect_and_write",
    "parse_args",
    "main",
]

SUMMARY_PATH_DEFAULT = "artifacts/pr_context.summary.txt"
APPENDIX_PATH_DEFAULT = "artifacts/pr_context.appendix.txt"
SUMMARY_CHAR_BUDGET = 16000
APPENDIX_CHAR_BUDGET = 48000
ISSUE_SUMMARY_LINE_BUDGET = 25
ISSUE_APPENDIX_LINE_BUDGET = 120
COMMENT_SUMMARY_LIMIT = 3
COMMENT_APPENDIX_LIMIT = 10
PR_BODY_SUMMARY_LINES = 25
PR_BODY_APPENDIX_LINES = 120


def _last_with_truncation(items: list[str], limit: int) -> tuple[list[str], bool]:
    if len(items) <= limit:
        return items, False
    return items[-limit:], True


def _extract_digest_bullets(body: str, *, headings: list[str], limit: int) -> list[str]:
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


def _issue_digest(issue: IssueDetails) -> str:
    bullets = _extract_digest_bullets(
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
        bullets = [
            f"State: {issue.state}",
            f"Labels: {', '.join(issue.labels) if issue.labels else '(none)'}",
        ]

    selected_comments, truncated = _last_with_truncation(
        issue.comments, COMMENT_SUMMARY_LIMIT
    )
    comment_block = (
        "\n".join(f"- {comment}" for comment in selected_comments)
        if selected_comments
        else "(no comments)"
    )
    if truncated:
        comment_block += f"\nTRUNCATED: last {COMMENT_SUMMARY_LIMIT} comments shown"

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


def _pr_digest(pr: PullRequestDetails) -> str:
    bullets = _extract_digest_bullets(
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


def _issue_appendix(issue: IssueDetails) -> str:
    body_text = truncate_lines(issue.body, ISSUE_APPENDIX_LINE_BUDGET)
    comments, truncated = _last_with_truncation(issue.comments, COMMENT_APPENDIX_LIMIT)
    comment_text = (
        "\n".join(f"- {c}" for c in comments) if comments else "(no comments)"
    )
    if truncated:
        comment_text += f"\nTRUNCATED: last {COMMENT_APPENDIX_LIMIT} comments shown"
    user_story_block = ""
    if issue.user_story_content:
        user_story_block = "\n".join(
            [
                "",
                f"User story ({issue.user_story_path or 'user-story.md'}):",
                truncate_lines(issue.user_story_content, ISSUE_APPENDIX_LINE_BUDGET),
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


def _pr_appendix(pr: PullRequestDetails) -> str:
    body_text = truncate_lines(pr.body, PR_BODY_APPENDIX_LINES)
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


def _parse_numstat_detailed(
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
        try:
            add_count = int(add_part) if add_part.isdigit() else 0
        except ValueError:
            add_count = 0
        try:
            del_count = int(del_part) if del_part.isdigit() else 0
        except ValueError:
            del_count = 0
        adds_total += add_count
        dels_total += del_count
        per_file[format_diff_path(file_part)] = (add_count, del_count)
    return adds_total, dels_total, per_file


def _parse_name_status_map(name_status_text: str) -> dict[str, str]:
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


def _is_scoping_doc(path: str) -> bool:
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


def _scoping_doc_changes(
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
    name_status_map = _parse_name_status_map(name_status_text)
    for path, status in name_status_map.items():
        if not _is_scoping_doc(path):
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


def write_output(text: str, out_path: Path, append: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        handle.write(text)


def collect_and_write(
    *,
    base: str | None,
    head: str | None,
    out: Path,
    appendix_out: Path | None,
    repo_root: Path,
    append: bool,
    include_untracked: bool,
) -> None:
    runner = SubprocessRunner()
    git = GitClient(runner, repo_root)
    resolved_root = git.resolve_root()
    git = GitClient(runner, resolved_root)
    gh = GhClient(runner, resolved_root)

    try:
        gh.ensure_available()
    except RuntimeError as exc:  # pragma: no cover - availability gate
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    summary_path = out
    appendix_path = appendix_out or Path(APPENDIX_PATH_DEFAULT)

    current_pr = gh.current_pr()

    context_result = build_pr_context(
        git=git,
        gh=gh,
        base_ref=base,
        head_ref=head,
        include_untracked=include_untracked,
        feature_issue_refs=[],
        current_pr=current_pr,
    )

    changed_paths = extract_changed_paths(context_result.text)
    feature_docs = gather_feature_excerpts(resolved_root, changed_paths)
    feature_issue_refs = sorted(
        {ref for doc in feature_docs for ref in doc.issue_refs if ref.strip()}
    )

    if feature_issue_refs:
        context_result = build_pr_context(
            git=git,
            gh=gh,
            base_ref=base,
            head_ref=head,
            include_untracked=include_untracked,
            feature_issue_refs=feature_issue_refs,
            current_pr=current_pr,
        )

    referenced_issues_set = set(context_result.referenced_issues)
    referenced_prs_set = set(context_result.referenced_prs)
    invalid_refs_set = set(context_result.invalid_references)
    branch_refs = extract_issue_references(git.branch_name())
    path_refs = extract_issue_references("\n".join(changed_paths))
    for ref in feature_issue_refs:
        formatted = ref if ref.startswith("#") else f"#{ref}"
        entity = gh.classify_entity(ref.lstrip("#"))
        if entity == "issue":
            referenced_issues_set.add(formatted)
        elif entity == "pull":
            referenced_prs_set.add(formatted)
        else:
            invalid_refs_set.add(formatted)
    for ref in branch_refs + path_refs:
        formatted = ref if ref.startswith("#") else f"#{ref}"
        entity = gh.classify_entity(ref.lstrip("#"))
        if entity == "issue":
            referenced_issues_set.add(formatted)
        elif entity == "pull":
            referenced_prs_set.add(formatted)
        else:
            invalid_refs_set.add(formatted)

    referenced_issues = sorted(referenced_issues_set)
    referenced_prs = sorted(referenced_prs_set)
    invalid_refs = sorted(invalid_refs_set)

    author_asserted: list[str] = []
    author_reason = "None (author has not asserted autoclose issues)"
    verified = context_result.verified_closing
    if current_pr is None:
        verified_reason = "None (no PR exists yet for this branch)"
    elif not verified:
        verified_reason = "None (closingIssuesReferences empty)"
    else:
        verified_reason = "(verified from GitHub PR metadata)"

    issues_to_fetch = sorted(set(verified + author_asserted + referenced_issues))
    issue_details: list[IssueDetails] = []
    for ref in issues_to_fetch:
        issue_details.append(gh.issue_details(ref.lstrip("#")))

    pr_details_list: list[PullRequestDetails] = []
    for ref in referenced_prs:
        pr_details_list.append(gh.pr_details(ref.lstrip("#")))

    if context_result.merge_base and context_result.head_sha:
        name_status_text = git.diff_range(
            ["--name-status", context_result.merge_base, context_result.head_sha]
        )
        numstat_text = git.diff_range(
            ["--numstat", context_result.merge_base, context_result.head_sha]
        )
    else:
        name_status_text = git.diff_range(["--name-status"])
        numstat_text = git.diff_range(["--numstat"])

    _additions, _deletions, per_file_stats = _parse_numstat_detailed(numstat_text)
    status_map = _parse_name_status_map(name_status_text)

    scoping_changes = _scoping_doc_changes(
        git=git,
        merge_base=context_result.merge_base,
        head_sha=context_result.head_sha,
        root=resolved_root,
        name_status_text=name_status_text,
        numstat_details=per_file_stats,
    )
    material_scoping = [
        (path, reasons, excerpt)
        for path, material, reasons, excerpt in scoping_changes
        if material
    ]
    non_material_scoping = [
        (path, reasons)
        for path, material, reasons, _ in scoping_changes
        if not material
    ]

    ci_target = context_result.head_sha
    if not ci_target:
        try:
            ci_target = git.rev_parse("HEAD")
        except RuntimeError:
            ci_target = None
    ci_status, ci_jobs = gh.ci_status(ci_target) if ci_target else (None, [])

    bucket_core: list[tuple[str, tuple[int, int]]] = []
    bucket_renames: list[tuple[str, tuple[int, int]]] = []
    bucket_docs: list[tuple[str, tuple[int, int]]] = []
    for path, status in status_map.items():
        stats = per_file_stats.get(path, (0, 0))
        if status.startswith("R"):
            bucket_renames.append((path, stats))
        elif path.endswith(".py") or path.endswith(".ps1"):
            bucket_core.append((path, stats))
        elif path.startswith("docs/") or path.startswith(".github") or "AGENTS" in path:
            bucket_docs.append((path, stats))

    def bucket_text(name: str, entries: list[tuple[str, tuple[int, int]]]) -> str:
        if not entries:
            return f"{name}: 0 files"
        sorted_entries = sorted(
            entries, key=lambda item: item[1][0] + item[1][1], reverse=True
        )
        lines = [
            f"{name}: {len(entries)} files",
            *(
                f"- {path} (+{adds}/-{dels})"
                for path, (adds, dels) in sorted_entries[:10]
            ),
        ]
        return "\n".join(lines)

    scoping_summary_lines: list[str] = []
    if material_scoping:
        scoping_summary_lines.append("Scoping docs changed (material):")
        for path, reasons, excerpt in material_scoping:
            reason_text = (
                f"Reasons: {', '.join(reasons) if reasons else '(unspecified)'}"
            )
            scoping_summary_lines.append(f"- {path} ({reason_text})")
            if excerpt:
                scoping_summary_lines.append(truncate_lines(excerpt, 40))
    if non_material_scoping:
        scoping_summary_lines.append("Scoping docs changed (non-material):")
        for path, reasons in non_material_scoping[:5]:
            reason_text = (
                f"Reasons: {', '.join(reasons) if reasons else '(unspecified)'}"
            )
            scoping_summary_lines.append(f"- {path} ({reason_text})")
    if not scoping_summary_lines:
        scoping_summary_lines.append("(none)")

    issue_digests = "\n\n".join(_issue_digest(detail) for detail in issue_details)
    pr_digests = "\n\n".join(_pr_digest(detail) for detail in pr_details_list)

    close_candidates = build_close_candidates_section(
        verified=verified,
        author_asserted=sorted(set(author_asserted)),
        referenced=referenced_issues,
        verified_reason=verified_reason,
        author_reason=author_reason,
    )

    gh_status_text = gh.status_message or "GitHub CLI authenticated."
    intent_block = "\n".join(
        [
            section("PR Intent"),
            "Primary outcome:",
            "User/dev impact:",
            "Risks:",
            "Author-asserted autoclose issues:",
        ]
    )

    summary_sections = [
        section("GitHub CLI status"),
        gh_status_text,
        intent_block,
        section("Base/Head"),
        f"Base ref (requested): {context_result.base_ref or '(default)'}",
        (
            f"Base ref (resolved): {context_result.resolved_base or '(unknown)'} @ "
            f"{context_result.base_sha or '(unknown)'}"
        ),
        (
            f"Head ref (resolved): {context_result.head_ref or head or '(unknown)'} @ "
            f"{context_result.head_sha or '(unknown)'}"
        ),
        f"Merge base: {context_result.merge_base or '(unknown)'}",
        f"Range: {context_result.rev_range or '(unknown)'}",
    ]
    if (
        context_result.base_ref
        and context_result.resolved_base
        and not str(context_result.resolved_base).startswith("origin/")
    ):
        summary_sections.append(
            "WARNING: Requested base is local and may be stale; prefer "
            f"origin/{context_result.base_ref}"
        )
    summary_sections.extend(
        [
            "",
            close_candidates,
            "",
            section("Referenced issues (classified)"),
            format_list(referenced_issues, "(none)"),
            "",
            section("PRs in range (classified)"),
            format_list(referenced_prs, "(none)"),
            "",
            section("Invalid references (not found)"),
            format_list(invalid_refs, "(none)"),
            "",
            section("Scoping docs changed"),
            "\n".join(scoping_summary_lines),
            "",
            section("Changed files overview"),
            bucket_text("Core logic changes", bucket_core),
            "",
            bucket_text("Mechanical moves/renames", bucket_renames),
            "",
            bucket_text("Docs/templates/agents/tooling", bucket_docs),
            "",
            section("Issue digests"),
            issue_digests or "(none)",
            "",
            section("PR digests"),
            pr_digests or "(none)",
            "",
            section("CI status (HEAD)"),
            (
                f"Status: {ci_status}\n"
                + (f"Failing jobs: {', '.join(ci_jobs)}" if ci_jobs else "")
                if ci_status
                else "(not available)"
            ),
            "",
            section("Appendix pointer"),
            f"See {appendix_path}",
        ]
    )

    summary_text = "\n".join(summary_sections)
    if len(summary_text) > SUMMARY_CHAR_BUDGET:
        summary_text = (
            summary_text[:SUMMARY_CHAR_BUDGET] + "\nTRUNCATED: summary budget exceeded"
        )

    feature_block = "\n".join(doc.excerpt for doc in feature_docs)

    issue_sections = [_issue_appendix(detail) for detail in issue_details]
    pr_sections = [_pr_appendix(detail) for detail in pr_details_list]
    appendix_parts = [
        section("Context generated")
        + "\n"
        + datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
        + "\n",
        context_result.text,
        "",
        section("Issue details"),
        "\n\n".join(issue_sections) if issue_sections else "(none)",
        "",
        section("Contributing pull requests"),
        "\n\n".join(pr_sections) if pr_sections else "(none)",
    ]
    if feature_block:
        appendix_parts.extend(["", feature_block])
    appendix_text = "\n".join(appendix_parts)
    if len(appendix_text) > APPENDIX_CHAR_BUDGET:
        appendix_text = (
            appendix_text[:APPENDIX_CHAR_BUDGET]
            + "\nTRUNCATED: appendix budget exceeded"
        )

    write_output(summary_text, summary_path, append)
    write_output(appendix_text, appendix_path, append)
    print(f"Wrote context summary to: {summary_path}")
    print(f"Wrote context appendix to: {appendix_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PR context for GitHub.")
    parser.add_argument(
        "--base", dest="base", help="Base ref (default: auto-detect origin/main)"
    )
    parser.add_argument("--head", dest="head", help="Head ref (default: HEAD)")
    parser.add_argument(
        "--out",
        dest="out",
        default=SUMMARY_PATH_DEFAULT,
        help="Summary output file path",
    )
    parser.add_argument(
        "--appendix-out",
        dest="appendix_out",
        default=APPENDIX_PATH_DEFAULT,
        help="Appendix output file path",
    )
    parser.add_argument(
        "--repo-root",
        dest="repo_root",
        default=".",
        help="Repository root (defaults to current directory)",
    )
    parser.add_argument(
        "--append",
        dest="append",
        action="store_true",
        help="Append instead of overwrite",
    )
    parser.add_argument(
        "--no-untracked",
        dest="no_untracked",
        action="store_true",
        help="Exclude untracked files from status",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.out).expanduser()
    repo_root = Path(args.repo_root).expanduser().resolve()
    collect_and_write(
        base=args.base,
        head=args.head,
        out=out_path,
        appendix_out=Path(args.appendix_out).expanduser(),
        repo_root=repo_root,
        append=bool(args.append),
        include_untracked=not bool(args.no_untracked),
    )


if __name__ == "__main__":
    main()

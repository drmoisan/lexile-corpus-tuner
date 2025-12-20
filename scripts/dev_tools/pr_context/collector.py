"""Collect Git repository context for pull request authorship."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from .feature_docs import (
    completed_plan_tasks,
    extract_issue_references,
    gather_feature_excerpts,
)
from .git import CommandRunner, GitClient, SubprocessRunner
from .github import GhClient
from .models import (
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
from .render import (
    build_close_candidates_section,
    build_pr_context,
    convert_numstat,
    extension_summary,
    extract_changed_paths,
    extract_merge_pr_numbers,
    format_diff_path,
    format_issue_details,
    format_pr_details,
    select_default_base,
)  # noqa: F401
from .summary_helpers import (
    append_generation_timestamp,
    bucket_text,
)
from .summary_helpers import (
    issue_appendix as _issue_appendix,
)
from .summary_helpers import (
    issue_digest as _issue_digest,
)
from .summary_helpers import (
    parse_name_status_map as _parse_name_status_map,
)
from .summary_helpers import (
    parse_numstat_detailed as _parse_numstat_detailed,
)
from .summary_helpers import (
    pr_appendix as _pr_appendix,
)
from .summary_helpers import (
    pr_digest as _pr_digest,
)
from .summary_helpers import (
    scoping_doc_changes as _scoping_doc_changes,
)

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


# helper functions moved to summary_helpers


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
    gh_available = True
    gh_status_override: str | None = None
    try:
        gh.ensure_available()
    except RuntimeError as exc:  # pragma: no cover - availability gate
        gh_available = False
        gh_status_override = f"GitHub CLI unavailable: {exc}"

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
        gh_available=gh_available,
    )

    changed_paths = extract_changed_paths(context_result.text)
    feature_docs = gather_feature_excerpts(resolved_root, changed_paths)
    additional_context_files = sorted(
        {path for doc in feature_docs for path in doc.context_files if path}
    )
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
            gh_available=gh_available,
        )

    referenced_issues_set = set(context_result.referenced_issues)
    referenced_prs_set = set(context_result.referenced_prs)
    invalid_refs_set = set(context_result.invalid_references)
    branch_refs = extract_issue_references(git.branch_name())
    path_refs = extract_issue_references("\n".join(changed_paths))
    if gh_available:
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
    else:
        referenced_issues_set.update(
            formatted if formatted.startswith("#") else f"#{formatted}"
            for formatted in feature_issue_refs
        )
        referenced_issues_set.update(
            formatted if formatted.startswith("#") else f"#{formatted}"
            for formatted in branch_refs + path_refs
        )

    referenced_issues = sorted(referenced_issues_set)
    referenced_prs = sorted(referenced_prs_set)
    invalid_refs = sorted(invalid_refs_set)

    author_asserted: list[str] = []
    author_reason = "None (author has not asserted autoclose issues)"
    verified = context_result.verified_closing if gh_available else []
    if not gh_available:
        verified_reason = "None (GitHub CLI unavailable)"
    elif current_pr is None:
        verified_reason = "None (no PR exists yet for this branch)"
    elif not verified:
        verified_reason = "None (closingIssuesReferences empty)"
    else:
        verified_reason = "(verified from GitHub PR metadata)"

    issues_to_fetch = sorted(set(verified + author_asserted + referenced_issues))
    issue_details: list[IssueDetails] = []
    if gh_available:
        for ref in issues_to_fetch:
            issue_details.append(gh.issue_details(ref.lstrip("#")))

    pr_details_list: list[PullRequestDetails] = []
    if gh_available:
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
    ci_status, ci_jobs = (
        gh.ci_status(ci_target) if (ci_target and gh_available) else (None, [])
    )

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

    feature_summary = (
        "\n\n".join(truncate_lines(doc.excerpt, 80) for doc in feature_docs)
        if feature_docs
        else "(none)"
    )

    close_candidates = build_close_candidates_section(
        verified=verified,
        author_asserted=sorted(set(author_asserted)),
        referenced=referenced_issues,
        verified_reason=verified_reason,
        author_reason=author_reason,
    )

    gh_status_text = (
        gh_status_override or gh.status_message or "GitHub CLI authenticated."
    )
    if not gh_available and not gh_status_override:
        gh_status_text = "GitHub CLI unavailable; references unverified."
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
            section("Additional context files"),
            format_list(additional_context_files, "(none)"),
            "",
            section("Feature doc excerpts"),
            feature_summary,
            "",
            section("Referenced issues (classified)"),
            format_list(referenced_issues, "(none)")
            + ("\nNOTE: Unverified (GitHub unavailable)" if not gh_available else ""),
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
        append_generation_timestamp(),
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

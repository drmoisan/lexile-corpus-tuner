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
    section,
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
    select_default_base,
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

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    header = section("Context generated") + "\n" + timestamp + "\n\n"

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

    referenced_issues = sorted(
        {ref for ref in context_result.referenced_issues + feature_issue_refs}
    )
    issue_sections: list[str] = []
    for ref in referenced_issues:
        try:
            details = gh.issue_details(ref.lstrip("#"))
            issue_sections.append(format_issue_details(details))
        except RuntimeError as exc:
            issue_sections.append(
                section(f"Issue {ref}") + f"(failed to fetch details: {exc})"
            )

    pr_sections: list[str] = []
    for ref in context_result.referenced_prs:
        try:
            details = gh.pr_details(ref.lstrip("#"))
            pr_sections.append(format_pr_details(details))
        except RuntimeError as exc:
            pr_sections.append(
                section(f"Pull Request {ref}") + f"(failed to fetch details: {exc})"
            )

    author_asserted = extract_issue_references(current_pr.body) if current_pr else []
    close_candidates = build_close_candidates_section(
        verified=context_result.verified_closing,
        author_asserted=sorted(set(author_asserted)),
        referenced=referenced_issues,
    )

    feature_block = "\n".join(doc.excerpt for doc in feature_docs)

    assembled_sections = [header, context_result.text, close_candidates]
    if issue_sections:
        assembled_sections.append(
            section("Issue details") + "\n" + "\n\n".join(issue_sections)
        )
    if pr_sections:
        assembled_sections.append(
            section("Contributing pull requests") + "\n" + "\n\n".join(pr_sections)
        )
    if feature_block:
        assembled_sections.append(feature_block)

    final_text = "\n".join(assembled_sections)

    write_output(final_text, out, append)
    print(f"Wrote context to: {out}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PR context for GitHub.")
    parser.add_argument(
        "--base", dest="base", help="Base ref (default: auto-detect origin/main)"
    )
    parser.add_argument("--head", dest="head", help="Head ref (default: HEAD)")
    parser.add_argument(
        "--out",
        dest="out",
        default="artifacts/pr_context.txt",
        help="Output file path",
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
        repo_root=repo_root,
        append=bool(args.append),
        include_untracked=not bool(args.no_untracked),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.dev_tools.collect_pr_context import (
    CommandResult,
    GitClient,
    IssueDetails,
    PullRequestDetails,
    _issue_appendix,  # pyright: ignore[reportPrivateUsage]
    _issue_digest,  # pyright: ignore[reportPrivateUsage]
    _last_with_truncation,  # pyright: ignore[reportPrivateUsage]
    _parse_name_status_map,  # pyright: ignore[reportPrivateUsage]
    _parse_numstat_detailed,  # pyright: ignore[reportPrivateUsage]
    _scoping_doc_changes,  # pyright: ignore[reportPrivateUsage]
    build_close_candidates_section,
    build_pr_context,
    convert_numstat,
    extension_summary,
    extract_issue_references,
    extract_merge_pr_numbers,
    find_user_story_link,
    format_diff_path,
    gather_feature_excerpts,
    select_default_base,
)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], CommandResult]) -> None:
        self.responses = responses

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult:
        key = tuple(args)
        return self.responses.get(key, CommandResult(stdout="", stderr="", code=1))


class FakeGit(GitClient):
    def __init__(self) -> None:
        super().__init__(FakeRunner({}), Path("."))
        self.called = []

    def branch_name(self) -> str:
        return "feature/test"

    def upstream(self) -> str:
        return "origin/feature/test"

    def remote_verbose(self) -> str:
        return "origin https://example/repo (fetch)"

    def status_short(self) -> str:
        return "## feature/test...origin/feature/test"

    def untracked(self) -> str:
        return "docs/features/active/fix-all-script/spec.md"

    def diff_name_status(self, *, staged: bool) -> str:
        marker = "A" if staged else "M"
        return f"{marker}\tdocs/features/active/fix-all-script/spec.md"

    def diff_patch(self, *, staged: bool) -> str:
        return "(diff omitted)"

    def rev_parse(self, ref: str) -> str:
        return "basehash" if ref == "main" else "headhash"

    def merge_base(self, base: str, head: str) -> str:
        return base

    def log(self, fmt: str, rev_range: str) -> str:
        if fmt.startswith("--pretty=format:"):
            return "\n".join(
                [
                    "a1 2025-01-01 alice Merge pull request #53",
                    "a2 2025-01-02 bob fix(scope): closes #44",
                ]
            )
        if fmt == "--pretty=%s":
            return "Merge pull request #53\nfix(scope): closes #44"
        if fmt == "--format=%an <%ae>":
            return "alice <a@example.com>\nbob <b@example.com>"
        return ""

    def diff_range(self, args: Sequence[str]) -> str:
        if "--name-status" in args:
            return "M\tdocs/features/active/fix-all-script/spec.md"
        if "--numstat" in args:
            return "4\t2\tdocs/features/active/fix-all-script/spec.md"
        if "--shortstat" in args:
            return " 1 files changed, 4 insertions(+), 2 deletions(-)"
        if "--stat" in args:
            return " spec.md | 6 +-"
        return ""

    def run(self, args: Sequence[str], *, allow_error: bool = False) -> CommandResult:
        return CommandResult(stdout="resolved", stderr="", code=0)


class FakeGh:
    def __init__(self) -> None:
        self.available = True

    def ensure_available(self) -> None:
        return None

    def classify_entity(self, number: str) -> str | None:
        if number == "44":
            return "issue"
        if number == "53":
            return "pull"
        return None


def test_format_diff_path_handles_brace_and_simple_renames():
    assert format_diff_path("dir/{old => new}/file.txt") == "dir/new/file.txt"
    assert format_diff_path("old.txt => new.txt") == "new.txt"
    assert format_diff_path('"quoted.txt"') == "quoted.txt"


def test_convert_numstat_sums_and_collects_files():
    adds, dels, files = convert_numstat("4\t2\tfile1.py\n-\t-\timage.png")
    assert adds == 4
    assert dels == 2
    assert files == ["file1.py", "image.png"]


def test_extension_summary_sorts_and_counts():
    summary = extension_summary(["a.py", "b.py", "Makefile"])
    lines = summary.splitlines()
    assert "py" in lines[1]
    assert "(noext)" in summary


def test_extract_issue_references_filters_and_deduplicates():
    text = "Fixes #12 and relates to ABC-99 plus #12 again"
    refs = extract_issue_references(text)
    assert refs == ["#12", "ABC-99"]


def test_last_with_truncation_limits_list():
    items, truncated = _last_with_truncation(["a", "b", "c", "d"], 2)
    assert items == ["c", "d"]
    assert truncated is True


def test_parse_name_status_map_collects_statuses():
    mapping = _parse_name_status_map("A\tfirst.txt\nM\tsecond.txt")
    assert mapping == {"first.txt": "A", "second.txt": "M"}


def test_extract_merge_pr_numbers_ignores_non_merge_lines():
    subjects = [
        "Merge pull request #51 from branch",
        "fix: close #5",
        "Merge pull request #51 from branch",  # duplicate should dedupe
    ]
    numbers = extract_merge_pr_numbers(subjects)
    assert numbers == ["#51"]


def test_select_default_base_tries_candidates_in_order():
    responses = {
        ("git", "rev-parse", "--verify", "--quiet", "origin/main"): CommandResult(
            "", "", 1
        ),
        ("git", "rev-parse", "--verify", "--quiet", "main"): CommandResult(
            "main", "", 0
        ),
    }
    runner = FakeRunner(responses)
    git = GitClient(runner, Path("."))
    assert select_default_base(git) == "main"


def test_build_pr_context_classifies_prs_and_issues_and_embeds_closing():
    git = FakeGit()
    gh = FakeGh()
    current_pr = PullRequestDetails(
        number="#99",
        state="open",
        author="alice",
        base_ref="main",
        head_ref="feature/test",
        created_at="2025-01-01",
        updated_at="2025-01-02",
        merged_at=None,
        labels=[],
        assignees=[],
        title="current",
        body="closes #44",
        closing_issues=["#50"],
        files_changed=[],
    )
    context = build_pr_context(
        git=git,
        gh=gh,
        base_ref="main",
        head_ref="feature/test",
        include_untracked=True,
        current_pr=current_pr,
    )
    assert "PRs in range" in context.text
    assert "#53" in context.text
    assert "Referenced issues" in context.text
    assert "#44" in context.text
    assert context.verified_closing == ["#50"]
    assert context.invalid_references == []


def test_gather_feature_excerpts_reads_active_docs():
    root = Path(__file__).resolve().parents[3]
    paths = [
        "docs/features/active/fix-all-script/spec.md",
        "docs/features/active/fix-all-script/plan.md",
    ]
    excerpts = gather_feature_excerpts(root, paths)
    joined = "\n".join(item.excerpt for item in excerpts)
    assert "fix-all-script" in joined
    assert "Spec excerpts" in joined
    assert "Plan completed tasks" in joined or "Spec excerpts" in joined
    collected_issue_refs = {ref for item in excerpts for ref in item.issue_refs}
    assert isinstance(collected_issue_refs, set)


def test_find_user_story_link_extracts_blob_path():
    link = find_user_story_link(
        "See [story](https://github.com/org/repo/blob/main/docs/story/user-story.md)"
    )
    assert link == "docs/story/user-story.md"


def test_build_close_candidates_section_renders_lists():
    section_text = build_close_candidates_section(
        verified=["#1"],
        author_asserted=["#2"],
        referenced=["#3", "#4"],
        verified_reason="None (no PR exists yet for this branch)",
        author_reason="None (author asserted)",
    )
    assert "Close candidates" in section_text
    assert "#1" in section_text and "#2" in section_text and "#3" in section_text


def test_issue_digest_truncates_comments():
    issue = IssueDetails(
        number="#10",
        title="Test",
        state="open",
        labels=["bug"],
        assignees=["alice"],
        author="bob",
        created_at="2024-01-01",
        updated_at="2024-01-02",
        body="## Why\n- reason one\n- reason two\n",
        comments=[f"comment {idx}" for idx in range(6)],
        user_story_path=None,
        user_story_content=None,
    )
    digest = _issue_digest(issue)
    assert "reason one" in digest
    assert "TRUNCATED: last 3 comments shown" in digest


def test_issue_appendix_truncates_body_and_comments():
    issue = IssueDetails(
        number="#11",
        title="Another",
        state="open",
        labels=[],
        assignees=[],
        author="carol",
        created_at="2024-02-01",
        updated_at="2024-02-02",
        body="\n".join(f"line {i}" for i in range(130)),
        comments=[f"note {i}" for i in range(15)],
        user_story_path=None,
        user_story_content=None,
    )
    appendix = _issue_appendix(issue)
    assert "TRUNCATED: first" in appendix or "TRUNCATED" in appendix
    assert "TRUNCATED: last 10 comments shown" in appendix


def test_parse_numstat_detailed_collects_per_file():
    adds, dels, mapping = _parse_numstat_detailed("5\t1\ta.py\n3\t2\tdocs/readme.md")
    assert adds == 8 and dels == 3
    assert mapping["a.py"] == (5, 1)
    assert mapping["docs/readme.md"] == (3, 2)


class FakeGitScoping(GitClient):
    def __init__(self, diff_text: str) -> None:
        super().__init__(FakeRunner({}), Path("."))
        self._diff_text = diff_text

    def diff_range(self, args: Sequence[str]) -> str:
        return self._diff_text


def test_scoping_doc_changes_flags_material_headings():
    root = Path(__file__).resolve().parents[3]
    path = "docs/features/active/fix-all-script/spec.md"
    diff_text = f"+++ b/{path}\n+## Acceptance Criteria\n+New criteria"
    git = FakeGitScoping(diff_text)
    changes = _scoping_doc_changes(
        git=git,
        merge_base="base",
        head_sha="head",
        root=root,
        name_status_text=f"M\t{path}",
        numstat_details={path: (10, 2)},
    )
    material = [entry for entry in changes if entry[1]]
    assert material


def test_scoping_doc_changes_marks_non_material_link_only():
    root = Path(__file__).resolve().parents[3]
    path = "docs/features/active/fix-all-script/spec.md"
    diff_text = f"+++ b/{path}\n+http://example.com\n+[link](https://example.com)"
    git = FakeGitScoping(diff_text)
    changes = _scoping_doc_changes(
        git=git,
        merge_base="base",
        head_sha="head",
        root=root,
        name_status_text=f"M\t{path}",
        numstat_details={path: (2, 0)},
    )
    assert changes
    assert changes[0][1] is False

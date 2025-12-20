from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

from scripts.dev_tools.pr_context.collector import (
    CommandResult,
    GitClient,
    IssueDetails,
    PullRequestDetails,
    build_close_candidates_section,
    build_pr_context,
    collect_and_write,
    convert_numstat,
    extension_summary,
    extract_issue_references,
    extract_merge_pr_numbers,
    find_user_story_link,
    format_diff_path,
    gather_feature_excerpts,
    main,
    parse_args,
    select_default_base,
    write_output,
)
from scripts.dev_tools.pr_context.models import FeatureDocExcerpt, PRContextResult
from scripts.dev_tools.pr_context.summary_helpers import (
    extract_digest_bullets as _extract_digest_bullets,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    is_scoping_doc as _is_scoping_doc,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    issue_appendix as _issue_appendix,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    issue_digest as _issue_digest,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    last_with_truncation as _last_with_truncation,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    parse_name_status_map as _parse_name_status_map,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    parse_numstat_detailed as _parse_numstat_detailed,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    pr_appendix as _pr_appendix,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    pr_digest as _pr_digest,
)
from scripts.dev_tools.pr_context.summary_helpers import (
    scoping_doc_changes as _scoping_doc_changes,
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
    assert "Author-asserted autoclose issues" in context.text
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
    for item in excerpts:
        assert any(
            context_path.endswith("spec.md") or context_path.endswith("plan.md")
            for context_path in item.context_files
        )


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


def test_pr_digest_and_appendix_cover_headings_and_files():
    pr = PullRequestDetails(
        number="#21",
        title="Improve docs",
        state="open",
        author="lee",
        base_ref="main",
        head_ref="feature/docs",
        created_at="2024-01-01",
        updated_at="2024-01-02",
        merged_at=None,
        labels=["docs"],
        assignees=["lee"],
        body="## Why\n- clarify usage\n",
        closing_issues=["#8"],
        files_changed=["a.md", "b.md", "c.md", "d.md"],
    )
    digest = _pr_digest(pr)
    appendix = _pr_appendix(pr)
    assert "Why: clarify usage" in digest
    assert "Files (first 25)" in appendix
    assert "closes" not in appendix.lower()


def test_issue_appendix_includes_user_story_block():
    issue = IssueDetails(
        number="#12",
        title="Story reference",
        state="open",
        labels=[],
        assignees=[],
        author="alex",
        created_at="2024-03-01",
        updated_at="2024-03-02",
        body="Context\n" + "\n".join(f"line {i}" for i in range(10)),
        comments=["note"],
        user_story_path="docs/story/user-story.md",
        user_story_content="Story content line 1\nline 2",
    )
    appendix = _issue_appendix(issue)
    assert "User story (docs/story/user-story.md)" in appendix
    assert "Story content line" in appendix


def test_parse_numstat_detailed_handles_non_numeric_entries():
    adds, dels, mapping = _parse_numstat_detailed(
        "-\t-\tfirst.txt\nnotnum\t3\tsecond.txt\n"
    )
    assert adds == 0
    assert dels == 3
    assert mapping["first.txt"] == (0, 0)
    assert mapping["second.txt"] == (0, 3)


def test_is_scoping_doc_identifies_feature_files():
    assert _is_scoping_doc("docs/features/active/feat/spec.md")
    assert _is_scoping_doc("docs/features/active/feat/plan.md")
    assert _is_scoping_doc("docs/features/active/feat/bug-remediation-plan.md")
    assert _is_scoping_doc("docs/features/active/feat/user-story.md")
    assert not _is_scoping_doc("docs/features/ideas/idea.md")
    assert not _is_scoping_doc("src/main.py")


def test_collect_and_write_uses_feature_refs_and_scoping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[tuple[Path, str]] = []

    def fake_write_output(text: str, out_path: Path, append: bool) -> None:
        captured.append((out_path, text))

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.write_output", fake_write_output
    )

    class FakeGit:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._root = Path(__file__).resolve().parents[3]

        def resolve_root(self) -> Path:
            return self._root

        def branch_name(self) -> str:
            return "feature/ABC-10"

        def diff_range(self, args: Sequence[str]) -> str:
            path = "docs/features/active/fix-all-script/spec.md"
            if "--name-status" in args:
                return f"M\t{path}"
            if "--numstat" in args:
                return f"20\t0\t{path}"
            if "--unified=0" in args:
                return f"+++ b/{path}\n+## Acceptance Criteria\n+New criteria"
            return ""

        def rev_parse(self, ref: str) -> str:
            return f"{ref}-sha"

    class FakeGh:
        status_message = "ok"

        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def ensure_available(self) -> None:
            return None

        def current_pr(self) -> PullRequestDetails | None:
            return None

        def classify_entity(self, number: str) -> str | None:
            return "issue" if number in {"1", "ABC-10"} else "pull"

        def issue_details(self, number: str) -> IssueDetails:
            return IssueDetails(
                number=f"#{number}",
                title="Issue",
                state="open",
                labels=[],
                assignees=[],
                author="alex",
                created_at="2024-01-01",
                updated_at="2024-01-02",
                body="Body",
                comments=[],
            )

        def pr_details(self, number: str) -> PullRequestDetails:
            return PullRequestDetails(
                number=f"#{number}",
                title="PR",
                state="open",
                author="alex",
                base_ref="main",
                head_ref="feature",
                created_at="2024-01-01",
                updated_at="2024-01-02",
                merged_at=None,
                labels=[],
                assignees=[],
                body="PR body",
                closing_issues=[],
                files_changed=["file.py"],
            )

        def ci_status(self, head_sha: str) -> tuple[str | None, list[str]]:
            return "success", []

    feature_calls: list[list[str]] = []

    def fake_build_pr_context(
        *,
        git: GitClient,
        gh: FakeGh,
        base_ref: str | None,
        head_ref: str | None,
        include_untracked: bool,
        feature_issue_refs: Sequence[str] | None = None,
        current_pr: PullRequestDetails | None = None,
        gh_available: bool | None = None,
    ) -> PRContextResult:
        feature_calls.append(list(feature_issue_refs or []))
        context_text = "\n".join(
            [
                "===== Changed files (name-status) =====",
                "M\tdocs/features/active/fix-all-script/spec.md",
                "===== Diff shortstat =====",
            ]
        )
        return PRContextResult(
            text=context_text,
            referenced_issues=["#1"],
            referenced_prs=["#2"],
            verified_closing=["#1"],
            invalid_references=[],
            base_ref=base_ref,
            resolved_base="origin/main",
            base_sha="base-sha",
            head_ref=head_ref or "feature",
            head_sha="head-sha",
            merge_base="base-sha",
            rev_range="base-sha..head-sha",
            gh_available=True,
        )

    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GitClient", FakeGit)
    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GhClient", FakeGh)
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.build_pr_context",
        fake_build_pr_context,
    )

    def fake_gather_feature_excerpts(
        root: Path, paths: Sequence[str]
    ) -> list[FeatureDocExcerpt]:
        return [
            FeatureDocExcerpt(
                feature="fix-all-script",
                excerpt="Excerpt",
                issue_refs=["#7"],
                context_files=["docs/features/active/fix-all-script/spec.md"],
            )
        ]

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.gather_feature_excerpts",
        fake_gather_feature_excerpts,
    )

    collect_and_write(
        base="main",
        head="feature",
        out=tmp_path / "summary.txt",
        appendix_out=tmp_path / "appendix.txt",
        repo_root=tmp_path,
        append=False,
        include_untracked=False,
    )

    assert feature_calls[0] == []
    assert feature_calls[1] == ["#7"]
    summary_text = next(text for path, text in captured if path.name == "summary.txt")
    appendix_text = next(text for path, text in captured if path.name == "appendix.txt")
    assert "Scoping docs changed" in summary_text
    assert "fix-all-script" in appendix_text


def test_extract_digest_bullets_truncates_and_skips_blank_lines() -> None:
    body = "## Why\n- first\n\n- second\n- third"
    bullets = _extract_digest_bullets(body, headings=["Why"], limit=2)
    assert bullets == ["Why: first", "Why: second"]


def test_parse_numstat_detailed_skips_invalid_rows() -> None:
    adds, dels, mapping = _parse_numstat_detailed("\n1\t1\tfile.py\ninvalid-entry")
    assert adds == 1 and dels == 1
    assert mapping == {"file.py": (1, 1)}


def test_write_output_creates_parent_and_appends(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.txt"
    write_output("first", target, append=False)
    write_output("second", target, append=True)
    assert target.read_text(encoding="utf-8").endswith("firstsecond")


def test_parse_args_and_main_delegate_to_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_collect_and_write(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.collect_and_write",
        fake_collect_and_write,
    )

    args = parse_args(
        [
            "--base",
            "origin/dev",
            "--head",
            "feature/123",
            "--out",
            "~/summary.txt",
            "--appendix-out",
            "~/appendix.txt",
            "--repo-root",
            ".",
            "--append",
            "--no-untracked",
        ]
    )
    assert args.base == "origin/dev"
    assert args.append is True

    main(
        [
            "--base",
            "origin/dev",
            "--head",
            "feature/123",
            "--out",
            "~/summary.txt",
            "--appendix-out",
            "~/appendix.txt",
            "--repo-root",
            ".",
            "--append",
            "--no-untracked",
        ]
    )

    assert captured["base"] == "origin/dev"
    assert captured["head"] == "feature/123"
    assert captured["append"] is True
    assert captured["include_untracked"] is False


def test_collect_and_write_renders_non_material_scoping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs: list[tuple[Path, str]] = []

    def fake_write_output(text: str, out_path: Path, append: bool) -> None:
        outputs.append((out_path, text))

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.write_output", fake_write_output
    )

    class StubGit:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._root = tmp_path

        def resolve_root(self) -> Path:
            return self._root

        def branch_name(self) -> str:
            return "feature/ABC-10"

        def upstream(self) -> str:
            return "origin/feature/ABC-10"

        def remote_verbose(self) -> str:
            return "origin https://example/repo (fetch)"

        def status_short(self) -> str:
            return "## feature/ABC-10"

        def untracked(self) -> str:
            return ""

        def diff_name_status(self, *, staged: bool) -> str:
            return ""

        def diff_patch(self, *, staged: bool) -> str:
            return ""

        def rev_parse(self, ref: str) -> str:
            return f"{ref}-sha"

        def merge_base(self, base: str, head: str) -> str:
            return "base-sha"

        def log(self, fmt: str, rev_range: str) -> str:
            return ""

        def diff_range(self, args: Sequence[str]) -> str:
            if "--name-status" in args:
                return "M\tdocs/features/active/feat/spec.md\nR100\told.py\tnew.py"
            if "--numstat" in args:
                return "2\t1\tdocs/features/active/feat/spec.md\n3\t0\tcore.py"
            if "--shortstat" in args:
                return " 1 files changed, 2 insertions(+), 1 deletions(-)"
            if "--stat" in args:
                return " spec.md | 3 ++"
            return ""

        def run(
            self, args: Sequence[str], *, allow_error: bool = False
        ) -> CommandResult:
            return CommandResult(stdout="resolved", stderr="", code=0)

    class StubGh:
        status_message = "ok"

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.available = True

        def ensure_available(self) -> None:
            return None

        def classify_entity(self, number: str) -> str | None:
            return "issue" if number == "1" else None

        def issue_details(self, number: str) -> IssueDetails:
            return IssueDetails(
                number=f"#{number}",
                title="Issue",
                state="open",
                labels=[],
                assignees=[],
                author="alex",
                created_at="2024-01-01",
                updated_at="2024-01-02",
                body="Body",
                comments=[],
            )

        def pr_details(self, number: str) -> PullRequestDetails:
            return PullRequestDetails(
                number=f"#{number}",
                title="PR",
                state="open",
                author="alex",
                base_ref="main",
                head_ref="feature",
                created_at="2024-01-01",
                updated_at="2024-01-02",
                merged_at=None,
                labels=[],
                assignees=[],
                body="PR body",
                closing_issues=["#1"],
                files_changed=["file.py"],
            )

        def ci_status(self, head_sha: str) -> tuple[str | None, list[str]]:
            return "success", []

        def current_pr(self) -> PullRequestDetails | None:
            return PullRequestDetails(
                number="#10",
                title="existing",
                state="open",
                author="alex",
                base_ref="main",
                head_ref="feature",
                created_at="2024-01-01",
                updated_at="2024-01-02",
                merged_at=None,
                labels=[],
                assignees=[],
                body="existing body",
                closing_issues=["#1"],
                files_changed=[],
            )

    def fake_build_pr_context(
        *,
        git: StubGit,
        gh: StubGh,
        base_ref: str | None,
        head_ref: str | None,
        include_untracked: bool,
        feature_issue_refs: Sequence[str] | None = None,
        current_pr: PullRequestDetails | None = None,
        gh_available: bool | None = None,
    ) -> PRContextResult:
        context_text = "\n".join(
            [
                "===== Changed files (name-status) =====",
                "M\tdocs/features/active/feat/spec.md",
                "A\tcore.py",
                "===== Diff shortstat =====",
                " 1 files changed, 2 insertions(+), 1 deletions(-)",
            ]
        )
        return PRContextResult(
            text=context_text,
            referenced_issues=["#1"],
            referenced_prs=["#2"],
            verified_closing=["#1"],
            invalid_references=["#999"],
            base_ref=base_ref,
            resolved_base="origin/main",
            base_sha="base-sha",
            head_ref=head_ref or "feature",
            head_sha="head-sha",
            merge_base="base-sha",
            rev_range="base-sha..head-sha",
            gh_available=True,
        )

    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GitClient", StubGit)
    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GhClient", StubGh)
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.build_pr_context", fake_build_pr_context
    )

    def fake_gather_feature_excerpts(
        *args: object, **kwargs: object
    ) -> list[FeatureDocExcerpt]:
        return []

    def fake_scoping_doc_changes(
        **kwargs: object,
    ) -> list[tuple[str, bool, list[str], None]]:
        return [("docs/features/active/feat/spec.md", False, ["links only"], None)]

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.gather_feature_excerpts",
        fake_gather_feature_excerpts,
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector._scoping_doc_changes",
        fake_scoping_doc_changes,
    )

    collect_and_write(
        base="main",
        head="feature",
        out=tmp_path / "summary.txt",
        appendix_out=tmp_path / "appendix.txt",
        repo_root=tmp_path,
        append=False,
        include_untracked=False,
    )

    summary_text = next(text for path, text in outputs if path.name == "summary.txt")
    assert "non-material" in summary_text
    assert "#999" in summary_text


def test_collect_and_write_handles_offline_gh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs: list[tuple[Path, str]] = []

    def fake_write_output(text: str, out_path: Path, append: bool) -> None:
        outputs.append((out_path, text))

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.write_output", fake_write_output
    )

    class OfflineGit:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._root = tmp_path

        def resolve_root(self) -> Path:
            return self._root

        def branch_name(self) -> str:
            return "feature/ABC-10"

        def upstream(self) -> str:
            return "origin/feature/ABC-10"

        def remote_verbose(self) -> str:
            return "origin https://example/repo (fetch)"

        def status_short(self) -> str:
            return "## feature/ABC-10"

        def untracked(self) -> str:
            return ""

        def diff_name_status(self, *, staged: bool) -> str:
            return ""

        def diff_patch(self, *, staged: bool) -> str:
            return ""

        def rev_parse(self, ref: str) -> str:
            return f"{ref}-sha"

        def merge_base(self, base: str, head: str) -> str:
            return "base-sha"

        def log(self, fmt: str, rev_range: str) -> str:
            return ""

        def diff_range(self, args: Sequence[str]) -> str:
            return ""

        def run(
            self, args: Sequence[str], *, allow_error: bool = False
        ) -> CommandResult:
            return CommandResult(stdout="resolved", stderr="", code=0)

    class OfflineGh:
        status_message = None
        available = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def ensure_available(self) -> None:
            raise RuntimeError("offline")

        def current_pr(self) -> PullRequestDetails | None:
            return None

        def classify_entity(self, number: str) -> str | None:
            return None

    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GitClient", OfflineGit)
    monkeypatch.setattr("scripts.dev_tools.pr_context.collector.GhClient", OfflineGh)

    collect_and_write(
        base="main",
        head="feature",
        out=tmp_path / "summary.txt",
        appendix_out=tmp_path / "appendix.txt",
        repo_root=tmp_path,
        append=False,
        include_untracked=False,
    )

    summary_text = next(text for path, text in outputs if path.name == "summary.txt")
    assert "GitHub CLI unavailable" in summary_text or "unavailable" in summary_text
    assert "Auto-close issues" in summary_text

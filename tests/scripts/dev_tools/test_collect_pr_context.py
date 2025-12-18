from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.dev_tools.collect_pr_context import (
    CommandResult,
    GhClient,
    GitClient,
    build_pr_context,
    completed_plan_tasks,
    convert_numstat,
    extension_summary,
    extract_issue_references,
    extract_merge_pr_numbers,
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


class FakeGh(GhClient):
    def __init__(self) -> None:
        self._available = True

    @property
    def available(self) -> bool:  # type: ignore[override]
        return self._available

    def classify_entity(self, number: str) -> str | None:
        if number == "44":
            return "issue"
        if number == "53":
            return "pull"
        return None

    def closing_issues(self, pr_number: str | None = None) -> list[str]:  # type: ignore[override]
        return ["#44", "#50"]


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
    context = build_pr_context(
        git=git,
        gh=gh,
        base_ref="main",
        head_ref="feature/test",
        include_untracked=True,
    )
    assert "PRs in range" in context
    assert "#53" in context
    assert "Referenced issues" in context
    assert "#44" in context
    assert "Issues to autoclose" in context
    assert "#50" in context


def test_gather_feature_excerpts_reads_active_docs():
    root = Path(__file__).resolve().parents[3]
    paths = [
        "docs/features/active/fix-all-script/spec.md",
        "docs/features/active/fix-all-script/plan.md",
    ]
    excerpts = gather_feature_excerpts(root, paths)
    joined = "\n".join(excerpts)
    assert "fix-all-script" in joined
    assert "Spec excerpts" in joined
    assert "Plan completed tasks" in joined or "Spec excerpts" in joined


def test_completed_plan_tasks_collects_checked_items():
    text = """
- [ ] todo
- [x] done item
- [X] another done
"""
    tasks = completed_plan_tasks(text)
    assert tasks == ["done item", "another done"]

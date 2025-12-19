from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scripts.dev_tools.pr_context_git import GitClient
from scripts.dev_tools.pr_context_models import CommandResult, PullRequestDetails
from scripts.dev_tools.pr_context_render import (
    build_pr_context,
    completed_plan_tasks,
    extract_changed_paths,
    parse_section,
    summarize_conventional_commits,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeRunner:
    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult:
        return CommandResult(stdout="", stderr="", code=1)


class FakeGitWarn(GitClient):
    def __init__(self) -> None:
        super().__init__(FakeRunner(), Path("."))

    def branch_name(self) -> str:
        return "feature/local"

    def upstream(self) -> str:
        return ""

    def remote_verbose(self) -> str:
        return "origin https://example/repo (fetch)"

    def status_short(self) -> str:
        return "## feature/local"

    def untracked(self) -> str:
        return ""

    def diff_name_status(self, *, staged: bool) -> str:
        return ""

    def diff_patch(self, *, staged: bool) -> str:
        return ""

    def run(self, args: Sequence[str], *, allow_error: bool = False) -> CommandResult:
        if "origin/local" in args:
            return CommandResult("", "", 1)
        return CommandResult("resolved", "", 0)

    def rev_parse(self, ref: str) -> str:
        return "basehash" if "local" in ref else "headhash"

    def merge_base(self, base: str, head: str) -> str:
        return "mergehash"

    def log(self, fmt: str, rev_range: str) -> str:
        if fmt.startswith("--pretty=format:"):
            return (
                "a1 2025-01-01 alice Merge pull request #53\n"
                "a2 2025-01-02 bob fix(scope): touches #999"
            )
        if fmt == "--pretty=%s":
            return "Merge pull request #53\nfix(scope): touches #999"
        if fmt == "--format=%an <%ae>":
            return "alice <a@example.com>\nbob <b@example.com>"
        return ""

    def diff_range(self, args: Sequence[str]) -> str:
        if "--name-status" in args:
            return "M\tsrc/main.py"
        if "--numstat" in args:
            return "5\t1\tsrc/main.py"
        if "--shortstat" in args:
            return " 1 file changed, 5 insertions(+), 1 deletion(-)"
        if "--stat" in args:
            return " src/main.py | 6 +-"
        return ""


class FakeGhClassify:
    def ensure_available(self) -> None:
        return None

    def classify_entity(self, number: str) -> str | None:
        if number == "53":
            return "pull"
        if number == "999":
            return None
        return "issue"


def test_build_pr_context_emits_base_warning_and_invalid_refs() -> None:
    git = FakeGitWarn()
    gh = FakeGhClassify()
    current_pr = PullRequestDetails(
        number="#1",
        title="existing",
        state="open",
        author="alex",
        base_ref="main",
        head_ref="feature/local",
        created_at="2024-01-01",
        updated_at="2024-01-02",
        merged_at=None,
        labels=[],
        assignees=[],
        body="",
        closing_issues=["#5"],
        files_changed=[],
    )
    context = build_pr_context(
        git=git,
        gh=gh,
        base_ref="local",
        head_ref=None,
        include_untracked=False,
        current_pr=current_pr,
    )
    assert "Base warning" in context.text
    assert "#999" in context.invalid_references
    assert "#53" in context.referenced_prs
    assert context.base_ref == "local"
    assert context.resolved_base == "origin/local" or context.resolved_base == "local"


def test_build_pr_context_handles_exceptions() -> None:
    class FailingGit(GitClient):
        def __init__(self) -> None:
            super().__init__(FakeRunner(), Path("."))

        def branch_name(self) -> str:
            return "feature/fail"

        def upstream(self) -> str:
            return "origin/feature/fail"

        def remote_verbose(self) -> str:
            return "origin https://example/repo (fetch)"

        def status_short(self) -> str:
            return "## feature/fail"

        def diff_name_status(self, *, staged: bool) -> str:
            return ""

        def diff_patch(self, *, staged: bool) -> str:
            return ""

        def rev_parse(self, ref: str) -> str:
            raise RuntimeError("boom")

    class DummyGh:
        def ensure_available(self) -> None:
            return None

        def classify_entity(self, number: str) -> str | None:
            return None

    git = FailingGit()
    gh = DummyGh()
    result = build_pr_context(
        git=git,
        gh=gh,
        base_ref="main",
        head_ref="feature/fail",
        include_untracked=False,
    )
    assert result.text.lstrip().startswith("===== PR Intent")
    assert result.referenced_issues == []
    assert result.resolved_base is None
    assert result.merge_base is None


def test_extract_changed_paths_and_helpers() -> None:
    context_text = "\n".join(
        [
            "===== Changed files (name-status) =====",
            "M\tsrc/app.py",
            "A\tdocs/guide.md",
            "===== Diff shortstat =====",
        ]
    )
    paths = extract_changed_paths(context_text)
    assert paths == ["src/app.py", "docs/guide.md"]

    section_text = parse_section("## Heading\ncontent\n## Next\n", "Heading")
    assert section_text == "content"
    assert parse_section("no heading", "Missing") == ""

    tasks = completed_plan_tasks("- [x] done 1\n- [X] done 2\n- [ ] skip\n")
    assert tasks == ["done 1", "done 2"]

    summary = summarize_conventional_commits("chore: clean\nfeature: skip\n")
    assert "other" in summary or "chore" in summary

    empty_summary = summarize_conventional_commits("\n\n")
    assert "(no recognizable conventional commit types)" in empty_summary

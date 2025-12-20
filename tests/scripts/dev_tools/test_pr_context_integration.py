from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.dev_tools.pr_context.collector import (
    FeatureDocExcerpt,
    IssueDetails,
    PRContextResult,
    PullRequestDetails,
    collect_and_write,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scripts.dev_tools.pr_context.git import CommandRunner, GitClient


class StubGit:
    def __init__(self, root: Path, changed_path: str) -> None:
        self._root = root
        self._changed_path = changed_path

    def resolve_root(self) -> Path:
        return self._root

    def branch_name(self) -> str:
        return "feature/docs"

    def upstream(self) -> str:
        return "origin/feature/docs"

    def remote_verbose(self) -> str:
        return "origin https://example/repo (fetch)"

    def status_short(self) -> str:
        return "## feature/docs"

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
            return f"M\t{self._changed_path}"
        if "--numstat" in args:
            return f"1\t0\t{self._changed_path}"
        return ""

    def run(self, args: Sequence[str], *, allow_error: bool = False):
        # The collector only expects command shape; stdout content is unused here.
        from scripts.dev_tools.pr_context.models import CommandResult

        return CommandResult(stdout="resolved", stderr="", code=0)


class OnlineGh:
    status_message = "ok"
    available = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    def ensure_available(self) -> None:
        return None

    def current_pr(self) -> PullRequestDetails | None:
        return None

    def classify_entity(self, number: str) -> str | None:
        if number.isdigit():
            return "issue"
        return None

    def ci_status(self, head_sha: str) -> tuple[str | None, list[str]]:
        return "success", []

    def issue_details(self, number: str) -> IssueDetails:
        return IssueDetails(
            number=f"#{number}",
            title=f"Issue {number}",
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
            head_ref="feature/docs",
            created_at="2024-01-01",
            updated_at="2024-01-02",
            merged_at=None,
            labels=[],
            assignees=[],
            body="PR body",
            closing_issues=[],
            files_changed=["file.py"],
        )


class OfflineGh(OnlineGh):
    status_message = None
    available = False

    def ensure_available(self) -> None:
        raise RuntimeError("offline")

    def classify_entity(self, number: str) -> str | None:
        return None


@pytest.mark.parametrize(
    (
        "gh_client_cls",
        "verified",
        "referenced",
        "feature_docs",
        "expect_autoclose",
        "expect_feature_block",
    ),
    [
        (
            OnlineGh,
            ["#42"],
            ["#5"],
            [
                FeatureDocExcerpt(
                    feature="2025-12-18-docs-v3-upgrade",
                    excerpt="Feature doc excerpt",
                    issue_refs=["#42", "#5"],
                    context_files=[
                        "docs/features/active/2025-12-18-docs-v3-upgrade/spec.md",
                        "docs/features/active/2025-12-18-docs-v3-upgrade/plan.md",
                    ],
                )
            ],
            "#42",
            True,
        ),
        (OfflineGh, [], ["#9"], [], None, False),
        (OnlineGh, [], [], [], None, False),
    ],
)
def test_collect_and_write_end_to_end_scenarios(
    gh_client_cls: type[OnlineGh],
    verified: list[str],
    referenced: list[str],
    feature_docs: list[FeatureDocExcerpt],
    expect_autoclose: str | None,
    expect_feature_block: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs: list[tuple[Path, str]] = []

    def fake_write_output(text: str, out_path: Path, append: bool) -> None:
        outputs.append((out_path, text))

    changed_path = "docs/features/active/2025-12-18-docs-v3-upgrade/spec.md"
    repo_root = Path(__file__).resolve().parents[3]

    def fake_build_pr_context(
        *,
        git: StubGit,
        gh: OnlineGh,
        base_ref: str | None,
        head_ref: str | None,
        include_untracked: bool,
        feature_issue_refs: Sequence[str] | None = None,
        current_pr: PullRequestDetails | None = None,
        gh_available: bool | None = None,
    ) -> PRContextResult:
        _ = feature_issue_refs, current_pr, gh_available, git, include_untracked
        context_text = "\n".join(
            [
                "===== Changed files (name-status) =====",
                f"M\t{changed_path}",
                "===== Diff shortstat =====",
                " 1 files changed, 1 insertions(+), 0 deletions(-)",
            ]
        )
        return PRContextResult(
            text=context_text,
            referenced_issues=referenced,
            referenced_prs=["#77"],
            verified_closing=verified,
            invalid_references=[],
            base_ref=base_ref,
            resolved_base="origin/main",
            base_sha="base-sha",
            head_ref=head_ref or "feature/docs",
            head_sha="head-sha",
            merge_base="base-sha",
            rev_range="base-sha..head-sha",
            gh_available=not isinstance(gh, OfflineGh),
        )

    def fake_git_client(runner: CommandRunner, root: Path) -> StubGit:
        _ = runner
        return StubGit(root, changed_path)

    def fake_feature_excerpts(
        root: Path, paths: Sequence[str]
    ) -> list[FeatureDocExcerpt]:
        _ = root, paths
        return feature_docs

    def fake_scoping_doc_changes(
        *,
        git: GitClient,
        merge_base: str | None,
        head_sha: str | None,
        root: Path,
        name_status_text: str,
        numstat_details: dict[str, tuple[int, int]],
    ) -> list[tuple[str, bool, list[str], str | None]]:
        _ = git, merge_base, head_sha, root, name_status_text, numstat_details
        return []

    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.write_output", fake_write_output
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.GitClient", fake_git_client
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.GhClient", gh_client_cls
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.build_pr_context", fake_build_pr_context
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector.gather_feature_excerpts",
        fake_feature_excerpts,
    )
    monkeypatch.setattr(
        "scripts.dev_tools.pr_context.collector._scoping_doc_changes",
        fake_scoping_doc_changes,
    )

    collect_and_write(
        base="main",
        head="feature/docs",
        out=tmp_path / "summary.txt",
        appendix_out=tmp_path / "appendix.txt",
        repo_root=repo_root,
        append=False,
        include_untracked=False,
    )

    summary_text = next(text for path, text in outputs if path.name == "summary.txt")
    appendix_text = next(text for path, text in outputs if path.name == "appendix.txt")

    if expect_autoclose:
        assert expect_autoclose in summary_text
    else:
        assert "Auto-close issues" in summary_text
        assert (
            "(none)" in summary_text
            or "None (author has not asserted autoclose issues)" in summary_text
        )

    if expect_feature_block:
        assert "Feature doc excerpts" in summary_text
        assert "Context files" in summary_text
        assert "Feature doc excerpt" in summary_text
        assert "Feature doc excerpt" in appendix_text
    else:
        assert "Feature doc excerpts" in summary_text
        assert "(none)" in summary_text or "Context files" in summary_text


def test_generate_pr_prompt_alignment() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "prompts"
        / "generate-pr.prompt.md"
    )
    content = prompt_path.read_text(encoding="utf-8")

    assert "using only" in content and "Additional context files" in content
    assert "Auto-close" in content
    assert (
        "None (GitHub validation unavailable; no verified closing issues listed)"
        in content
    )
    assert "Related issues / PRs" in content

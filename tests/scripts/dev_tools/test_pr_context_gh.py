from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import pytest

from scripts.dev_tools.pr_context.github import GhClient
from scripts.dev_tools.pr_context.models import CommandResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


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
        return self.responses.get(tuple(args), CommandResult("", "", 1))


def test_gh_client_classifies_entities_and_reports_status(tmp_path: Path) -> None:
    repo_json = '{"nameWithOwner": "owner/repo"}'
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            repo_json, "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/10"): CommandResult(
            '{"pull_request": {"url": "x"}}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/5"): CommandResult(
            '{"title": "issue"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/999"): CommandResult("", "Not Found", 1),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    gh.ensure_available()
    assert gh.status_message and "owner/repo" in gh.status_message
    assert gh.classify_entity("10") == "pull"
    assert gh.classify_entity("5") == "issue"
    assert gh.classify_entity("999") is None


def test_issue_and_pr_details_parse_metadata_and_comments(tmp_path: Path) -> None:
    story_path = "docs/story/user-story.md"
    story_b64 = base64.b64encode(b"story content").decode("ascii")
    issue_payload = f"""
{{
  "title": "Issue title",
  "state": "open",
  "body": "See ({story_path})",
  "labels": [{{"name": "bug"}}],
  "assignees": [{{"login": "alex"}}],
  "user": {{"login": "author"}},
  "created_at": "2024-01-01",
  "updated_at": "2024-01-02",
  "comments_url": "comments"
}}
"""
    pr_payload = """
{
  "number": 3,
  "title": "PR title",
  "state": "open",
  "body": "Body",
  "author": {"login": "taylor"},
  "baseRefName": "main",
  "headRefName": "feature",
  "createdAt": "2024-02-01",
  "updatedAt": "2024-02-02",
  "mergedAt": null,
  "labels": [{"name": "enhancement"}],
  "assignees": [{"login": "alex"}],
  "closingIssuesReferences": [{"number": 5}],
  "files": [{"path": "file1.py"}, {"path": "docs/readme.md"}]
}
"""
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/7"): CommandResult(issue_payload, "", 0),
        (
            "gh",
            "api",
            "comments?per_page=50",
        ): CommandResult(
            json.dumps(
                [
                    {
                        "user": {"login": "reviewer"},
                        "body": "Looks good",
                        "created_at": "2024-01-03",
                    }
                ]
            ),
            "",
            0,
        ),
        (
            "gh",
            "api",
            "repos/owner/repo/contents/docs/story/user-story.md",
        ): CommandResult(
            f'{{"content": "{story_b64}"}}',
            "",
            0,
        ),
        (
            "gh",
            "pr",
            "view",
            "3",
            "--json",
            "number,title,body,state,author,baseRefName,headRefName,createdAt,updatedAt,mergedAt,labels,assignees,closingIssuesReferences,files",
        ): CommandResult(pr_payload, "", 0),
        (
            "gh",
            "run",
            "list",
            "--commit",
            "abc123",
            "--limit",
            "1",
            "--json",
            "conclusion,status,name,headSha",
        ): CommandResult('[{"status": "success"}]', "", 0),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    issue = gh.issue_details("7")
    assert issue.labels == ["bug"]
    assert "reviewer" in issue.comments[0]
    assert issue.user_story_content == "story content"

    pr = gh.pr_details("3")
    assert pr.closing_issues == ["#5"]
    assert pr.labels == ["enhancement"]
    assert pr.files_changed == ["file1.py", "docs/readme.md"]

    status, failing = gh.ci_status("abc123")
    assert status == "success"
    assert failing == []


def test_gh_client_reports_unavailable_and_allows_not_found(tmp_path: Path) -> None:
    responses = {
        ("gh", "auth", "status"): CommandResult("", "not logged in", 1),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    with pytest.raises(RuntimeError):
        gh.ensure_available()

    responses_ok = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/999"): CommandResult("", "Not Found", 1),
    }
    gh_ok = GhClient(FakeRunner(responses_ok), tmp_path, gh_path="gh")
    assert gh_ok.classify_entity("999") is None


def test_current_pr_parses_payload_and_ci_status_handles_empty(tmp_path: Path) -> None:
    pr_payload = json.dumps(
        {
            "number": 4,
            "title": "Existing PR",
            "body": "Body",
            "state": "open",
            "author": {"login": "alex"},
            "baseRefName": "main",
            "headRefName": "feature",
            "createdAt": "2024-02-01",
            "updatedAt": "2024-02-02",
            "mergedAt": None,
            "labels": [{"name": "bug"}],
            "assignees": [{"login": "lee"}],
            "closingIssuesReferences": [{"number": 7}, {"number": 7}],
        }
    )
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        (
            "gh",
            "pr",
            "view",
            "--json",
            "number,title,body,state,author,baseRefName,headRefName,createdAt,updatedAt,mergedAt,labels,assignees,closingIssuesReferences",
        ): CommandResult(pr_payload, "", 0),
        (
            "gh",
            "run",
            "list",
            "--commit",
            "deadbeef",
            "--limit",
            "1",
            "--json",
            "conclusion,status,name,headSha",
        ): CommandResult("[]", "", 0),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    pr = gh.current_pr()
    assert pr is not None
    assert pr.closing_issues == ["#7"]

    status, jobs = gh.ci_status("deadbeef")
    assert status is None
    assert jobs == []


def test_gh_client_handles_unavailable_gh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_gh(_: str) -> None:
        return None

    monkeypatch.setattr("scripts.dev_tools.pr_context.github.shutil.which", missing_gh)
    gh = GhClient(FakeRunner({}), tmp_path, gh_path=None)
    with pytest.raises(RuntimeError):
        gh.ensure_available()
    assert gh.status_message is None


def test_gh_client_reports_authentication_failure(tmp_path: Path) -> None:
    responses = {
        ("gh", "auth", "status"): CommandResult("", "denied", 1),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    with pytest.raises(RuntimeError):
        gh.ensure_available()
    assert gh.available is False


def test_classify_entity_returns_none_on_not_found(tmp_path: Path) -> None:
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/99"): CommandResult("", "Not Found", 1),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    assert gh.classify_entity("99") is None


def test_closing_issues_parses_numbers(tmp_path: Path) -> None:
    payload = json.dumps({"closingIssuesReferences": [{"number": 4}, {"number": 5}]})
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "pr", "view", "--json", "closingIssuesReferences,number"): CommandResult(
            payload, "", 0
        ),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    assert gh.closing_issues() == ["#4", "#5"]


def test_issue_details_reads_local_story(tmp_path: Path) -> None:
    story_path = tmp_path / "docs/story/user-story.md"
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text("local story", encoding="utf-8")
    issue_payload = json.dumps(
        {
            "title": "Issue title",
            "state": "open",
            "body": f"See ({story_path.relative_to(tmp_path)})",
            "labels": [],
            "assignees": [],
            "user": {"login": "author"},
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
            "comments_url": None,
        }
    )
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/issues/8"): CommandResult(issue_payload, "", 0),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    details = gh.issue_details("8")
    assert details.user_story_content == "local story"


def test_fetch_repo_file_returns_none_on_invalid_payload(tmp_path: Path) -> None:
    responses = {
        ("gh", "auth", "status"): CommandResult("", "", 0),
        ("gh", "repo", "view", "--json", "nameWithOwner"): CommandResult(
            '{"nameWithOwner": "owner/repo"}', "", 0
        ),
        ("gh", "api", "repos/owner/repo/contents/path"): CommandResult("[]", "", 0),
    }
    gh = GhClient(FakeRunner(responses), tmp_path, gh_path="gh")
    assert gh.fetch_repo_file("path") is None

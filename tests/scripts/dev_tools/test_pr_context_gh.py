from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

from scripts.dev_tools.pr_context_gh import GhClient
from scripts.dev_tools.pr_context_models import CommandResult

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

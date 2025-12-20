"""Unit tests for scripts.dev_tools.pr_context.github module.

Focused on driving coverage of key uncovered paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.dev_tools.pr_context.git import CommandRunner
from scripts.dev_tools.pr_context.github import GhClient
from scripts.dev_tools.pr_context.models import CommandResult


class TestGhClientAvailability:
    """Test GhClient availability checks."""

    def test_gh_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GhClient marks unavailable when gh binary not found."""
        monkeypatch.setattr("shutil.which", lambda x: None)  # type: ignore[arg-type]
        runner = Mock(spec=CommandRunner)

        client = GhClient(runner, tmp_path, gh_path=None)

        assert not client.available
        # When unavailable, status_message is None
        assert client.status_message is None

    def test_gh_not_authenticated(self, tmp_path: Path) -> None:
        """GhClient marks unavailable when auth fails."""
        runner = Mock(spec=CommandRunner)
        runner.run.return_value = CommandResult("", "not logged in", 1)

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        assert not client.available
        # When unavailable, status_message is None
        assert client.status_message is None

    def test_gh_available_success(self, tmp_path: Path) -> None:
        """GhClient marks available when fully configured."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),  # auth status
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),  # repo view
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        assert client.available
        assert client.status_message == "GitHub CLI authenticated for owner/repo"

    def test_ensure_available_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_available raises when not available."""
        monkeypatch.setattr("shutil.which", lambda x: None)  # type: ignore[arg-type]
        runner = Mock(spec=CommandRunner)

        client = GhClient(runner, tmp_path, gh_path=None)

        with pytest.raises(RuntimeError):
            client.ensure_available()

    def test_repo_name_caching(self, tmp_path: Path) -> None:
        """status_message caches repo name."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        # Status message includes repo name
        msg = client.status_message
        assert msg and "owner/repo" in msg

    def test_repo_name_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """GhClient handles invalid JSON from repo view."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),  # auth status
            CommandResult("invalid", "", 0),  # Invalid JSON from repo view (init)
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        # Client should mark unavailable when repo resolution fails
        assert not client.available


class TestGhClientClassifyEntity:
    """Test entity classification."""

    def test_classify_as_issue(self, tmp_path: Path) -> None:
        """classify_entity returns 'issue' for issues."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult('{"number": 1}', "", 0),  # classify call
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        result = client.classify_entity("1")

        assert result == "issue"

    def test_classify_as_pull(self, tmp_path: Path) -> None:
        """classify_entity returns 'pull' when pull_request key present."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult('{"number": 1, "pull_request": {}}', "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        result = client.classify_entity("1")

        assert result == "pull"

    def test_classify_not_found(self, tmp_path: Path) -> None:
        """classify_entity returns None when entity not found."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult("", "404 Not Found", 1),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        result = client.classify_entity("999")

        assert result is None


class TestGhClientClosingIssues:
    """Test closing_issues method."""

    def test_closing_issues_returns_list(self, tmp_path: Path) -> None:
        """closing_issues extracts issue numbers."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps({"closingIssuesReferences": [{"number": 5}]}), "", 0
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        issues = client.closing_issues()

        assert issues == ["#5"]

    def test_closing_issues_empty(self, tmp_path: Path) -> None:
        """closing_issues returns empty list when no issues."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"closingIssuesReferences": []}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        issues = client.closing_issues()

        assert issues == []


class TestGhClientIssueDetails:
    """Test issue_details method."""

    def test_issue_details_minimal(self, tmp_path: Path) -> None:
        """issue_details handles minimal payload."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"number": 1}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.number == "#1"
        assert details.title == "(no title)"

    def test_issue_details_with_labels(self, tmp_path: Path) -> None:
        """issue_details extracts labels."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps({"number": 1, "labels": [{"name": "bug"}]}), "", 0
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.labels == ["bug"]


class TestGhClientPrDetails:
    """Test pr_details method."""

    def test_pr_details_basic(self, tmp_path: Path) -> None:
        """pr_details returns PullRequestDetails."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"number": 1, "title": "Test"}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.number == "#1"
        assert details.title == "Test"


class TestGhClientCurrentPr:
    """Test current_pr method."""

    def test_current_pr_not_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """current_pr returns None when gh not available."""
        monkeypatch.setattr("shutil.which", lambda x: None)  # type: ignore[arg-type]
        runner = Mock(spec=CommandRunner)

        client = GhClient(runner, tmp_path, gh_path=None)
        pr = client.current_pr()

        assert pr is None

    def test_current_pr_no_pr(self, tmp_path: Path) -> None:
        """current_pr returns None when no PR active."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult("", "no pull request", 1),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is None


class TestGhClientFetchRepoFile:
    """Test fetch_repo_file method."""

    def test_fetch_repo_file_success(self, tmp_path: Path) -> None:
        """fetch_repo_file decodes content."""
        import base64

        content = "test content"
        encoded = base64.b64encode(content.encode()).decode()

        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"content": encoded}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        result = client.fetch_repo_file("test.txt")

        assert result == content

    def test_fetch_repo_file_not_found(self, tmp_path: Path) -> None:
        """fetch_repo_file returns None when file not found."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult("", "404", 1),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        result = client.fetch_repo_file("missing.txt")

        assert result is None


class TestGhClientCiStatus:
    """Test ci_status method."""

    def test_ci_status_returns_status(self, tmp_path: Path) -> None:
        """ci_status extracts status from runs."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps([{"status": "success"}]), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        status, _ = client.ci_status("abc123")

        assert status == "success"

    def test_ci_status_empty_runs(self, tmp_path: Path) -> None:
        """ci_status returns None when no runs."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps([]), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        status, _ = client.ci_status("abc123")

        assert status is None


class TestGhClientIssueDetailsExtended:
    """Extended tests for issue_details covering comments and edge cases."""

    def test_issue_details_with_comments(self, tmp_path: Path) -> None:
        """issue_details extracts comments."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "title": "Test",
                        "comments_url": "https://api.github.com/repos/owner/repo/issues/1/comments",
                    }
                ),
                "",
                0,
            ),
            CommandResult(
                json.dumps(
                    [
                        {
                            "user": {"login": "testuser"},
                            "created_at": "2024-01-01T00:00:00Z",
                            "body": "Test comment",
                        }
                    ]
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert len(details.comments) == 1
        assert "testuser" in details.comments[0]
        assert "Test comment" in details.comments[0]

    def test_issue_details_with_body(self, tmp_path: Path) -> None:
        """issue_details extracts body text."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"number": 1, "body": "Issue body text"}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.body == "Issue body text"

    def test_issue_details_with_assignees(self, tmp_path: Path) -> None:
        """issue_details extracts assignees."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps({"number": 1, "assignees": [{"login": "user1"}]}), "", 0
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.assignees == ["user1"]


class TestGhClientPrDetailsExtended:
    """Extended tests for pr_details."""

    def test_pr_details_with_all_fields(self, tmp_path: Path) -> None:
        """pr_details extracts all available fields."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "title": "Test PR",
                        "body": "PR body",
                        "state": "open",
                        "headRefName": "feature-branch",
                        "baseRefName": "main",
                        "labels": [{"name": "bug"}],
                        "assignees": [{"login": "dev1"}],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.number == "#1"
        assert details.title == "Test PR"
        assert details.body == "PR body"
        assert details.state == "open"
        assert details.head_ref == "feature-branch"
        assert details.base_ref == "main"
        assert details.labels == ["bug"]
        assert details.assignees == ["dev1"]


class TestGhClientCurrentPrExtended:
    """Extended tests for current_pr."""

    def test_current_pr_success(self, tmp_path: Path) -> None:
        """current_pr returns PR number when active."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(json.dumps({"number": 42}), "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is not None
        assert pr.number == "#42"

    def test_current_pr_invalid_json(self, tmp_path: Path) -> None:
        """current_pr returns None on invalid JSON."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult("invalid", "", 0),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is None

    def test_current_pr_with_labels_and_assignees(self, tmp_path: Path) -> None:
        """current_pr extracts labels and assignees."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 42,
                        "title": "Current PR",
                        "labels": [{"name": "feature"}],
                        "assignees": [{"login": "dev1"}],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is not None
        assert pr.labels == ["feature"]
        assert pr.assignees == ["dev1"]

    def test_current_pr_with_closing_issues(self, tmp_path: Path) -> None:
        """current_pr extracts closing issues."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 42,
                        "closingIssuesReferences": [{"number": 10}],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is not None
        assert pr.closing_issues == ["#10"]

    def test_current_pr_with_author(self, tmp_path: Path) -> None:
        """current_pr extracts author."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 42,
                        "author": {"login": "contributor1"},
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is not None
        assert pr.author == "contributor1"

    def test_current_pr_malformed_data(self, tmp_path: Path) -> None:
        """current_pr handles malformed lists gracefully."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 42,
                        "labels": ["not a dict", {"name": "bug"}],
                        "assignees": ["not a dict", {"login": "dev2"}],
                        "closingIssuesReferences": ["not a dict", {"number": 5}],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        pr = client.current_pr()

        assert pr is not None
        assert pr.labels == ["bug"]
        assert pr.assignees == ["dev2"]
        assert pr.closing_issues == ["#5"]


class TestGhClientUserStory:
    """Test user story fetching in issue_details."""

    def test_issue_details_with_user_story_link(self, tmp_path: Path) -> None:
        """issue_details extracts user story link from body."""
        # Create a local user story file
        story_path = (
            tmp_path / "docs" / "features" / "active" / "test" / "user-story.md"
        )
        story_path.parent.mkdir(parents=True)
        story_path.write_text("# User Story Content", encoding="utf-8")

        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "title": "Feature request",
                        "body": "## User Story\n[user-story.md](docs/features/active/test/user-story.md)",
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.user_story_path == "docs/features/active/test/user-story.md"
        assert details.user_story_content == "# User Story Content"

    def test_issue_details_user_story_remote_fetch(self, tmp_path: Path) -> None:
        """issue_details fetches user story from remote when not local."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "title": "Feature",
                        "body": "[user-story.md](docs/features/active/test/user-story.md)",
                    }
                ),
                "",
                0,
            ),
            CommandResult(
                json.dumps(
                    {"content": "IyBSZW1vdGUgVXNlciBTdG9yeQ=="}
                ),  # base64 "# Remote User Story"
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.user_story_path == "docs/features/active/test/user-story.md"
        assert details.user_story_content == "# Remote User Story"


class TestGhClientPrDetailsError:
    """Test error handling in pr_details."""

    def test_pr_details_raises_on_no_repo(self, tmp_path: Path) -> None:
        """pr_details raises when repo unavailable."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult("invalid", "", 0),  # Bad JSON
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        with pytest.raises(RuntimeError, match="failed to resolve repository"):
            client.pr_details("1")

    def test_pr_details_raises_on_bad_payload(self, tmp_path: Path) -> None:
        """pr_details raises on non-dict payload."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult("[]", "", 0),  # Array instead of object
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")

        with pytest.raises(RuntimeError, match="Unexpected pull request payload"):
            client.pr_details("1")


class TestGhClientFilesChanged:
    """Test files_changed extraction in pr_details."""

    def test_pr_details_extracts_files(self, tmp_path: Path) -> None:
        """pr_details extracts files changed list."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "files": [
                            {"path": "src/main.py"},
                            {"path": "tests/test_main.py"},
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.files_changed == ["src/main.py", "tests/test_main.py"]


class TestGhClientCommentEdgeCases:
    """Test edge cases in comment extraction."""

    def test_issue_details_comment_without_user(self, tmp_path: Path) -> None:
        """issue_details handles comments without user field."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "comments_url": "https://api.github.com/repos/owner/repo/issues/1/comments",
                    }
                ),
                "",
                0,
            ),
            CommandResult(
                json.dumps(
                    [
                        {
                            "body": "Comment without user",
                            "created_at": "2024-01-01T00:00:00Z",
                        }
                    ]
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert len(details.comments) == 1
        assert "(unknown)" in details.comments[0]

    def test_issue_details_comment_malformed_entries(self, tmp_path: Path) -> None:
        """issue_details skips malformed comment entries."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "comments_url": "https://api.github.com/repos/owner/repo/issues/1/comments",
                    }
                ),
                "",
                0,
            ),
            CommandResult(
                json.dumps(
                    [
                        "not a dict",  # Should be skipped
                        {"body": "Good comment", "user": {"login": "user1"}},
                    ]
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert len(details.comments) == 1
        assert "user1" in details.comments[0]


class TestGhClientLabelAssigneeEdgeCases:
    """Test edge cases in label/assignee extraction."""

    def test_pr_details_malformed_labels(self, tmp_path: Path) -> None:
        """pr_details handles malformed label entries."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "labels": [
                            "not a dict",  # Should be skipped
                            {"name": "bug"},
                            {"no_name_field": "value"},  # Should be skipped
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.labels == ["bug"]

    def test_pr_details_malformed_assignees(self, tmp_path: Path) -> None:
        """pr_details handles malformed assignee entries."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "assignees": [
                            "not a dict",
                            {"login": "dev1"},
                            {"no_login_field": "value"},
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.assignees == ["dev1"]

    def test_pr_details_closing_issues_malformed(self, tmp_path: Path) -> None:
        """pr_details handles malformed closing issues."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "closingIssuesReferences": [
                            "not a dict",
                            {"number": 5},
                            {"no_number": "value"},
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.closing_issues == ["#5"]

    def test_pr_details_author_extraction(self, tmp_path: Path) -> None:
        """pr_details extracts author login."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "author": {"login": "contributor1"},
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.pr_details("1")

        assert details.author == "contributor1"

    def test_issue_details_malformed_labels(self, tmp_path: Path) -> None:
        """issue_details handles malformed label entries."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "labels": [
                            "not a dict",
                            {"name": "bug"},
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.labels == ["bug"]

    def test_issue_details_malformed_assignees(self, tmp_path: Path) -> None:
        """issue_details handles malformed assignee entries."""
        runner = Mock(spec=CommandRunner)
        runner.run.side_effect = [
            CommandResult("Logged in", "", 0),
            CommandResult('{"nameWithOwner": "owner/repo"}', "", 0),
            CommandResult(
                json.dumps(
                    {
                        "number": 1,
                        "assignees": [
                            "not a dict",
                            {"login": "assignee1"},
                        ],
                    }
                ),
                "",
                0,
            ),
        ]

        client = GhClient(runner, tmp_path, gh_path="/usr/bin/gh")
        details = client.issue_details("1")

        assert details.assignees == ["assignee1"]

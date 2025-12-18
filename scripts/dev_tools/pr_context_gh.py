"""GitHub CLI helpers for PR context collection."""

from __future__ import annotations

import base64
import json
import shutil
from typing import TYPE_CHECKING, cast

from .pr_context_models import IssueDetails, PullRequestDetails, find_user_story_link

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .pr_context_git import CommandRunner


class GhClient:
    """GitHub CLI wrapper used for entity classification and content fetches."""

    def __init__(
        self, runner: CommandRunner, cwd: Path, gh_path: str | None = None
    ) -> None:
        self._runner = runner
        self._cwd = cwd
        self._gh_path = gh_path or shutil.which("gh")
        self._repo_cache: str | None = None
        self._availability_error: str | None = None
        self._available = False
        self._hydrate_availability()

    @property
    def available(self) -> bool:
        return self._available

    def ensure_available(self) -> None:
        if not self._available:
            message = self._availability_error or "GitHub CLI is unavailable."
            raise RuntimeError(message)

    def _hydrate_availability(self) -> None:
        if not self._gh_path:
            self._availability_error = "GitHub CLI (gh) is not installed. Install from https://cli.github.com/."
            return

        status = self._runner.run(
            [self._gh_path, "auth", "status"], cwd=self._cwd, allow_error=True
        )
        if status.code != 0:
            stderr = status.stderr or status.stdout or "unknown error"
            self._availability_error = (
                "GitHub CLI is installed but not authenticated. Run 'gh auth login' "
                "to authenticate."
            )
            if stderr:
                self._availability_error += f" Details: {stderr.strip()}"
            return

        repo = self._repo_name()
        if not repo:
            self._availability_error = (
                "GitHub CLI is authenticated but failed to resolve repository. "
                "Ensure network access and repository permissions."
            )
            return

        self._available = True
        self._availability_error = None

    def _repo_name(self) -> str | None:
        if self._repo_cache:
            return self._repo_cache
        if not self._gh_path:
            return None

        result = self._runner.run(
            [self._gh_path, "repo", "view", "--json", "nameWithOwner"],
            cwd=self._cwd,
            allow_error=True,
        )
        if result.code != 0 or not result.stdout:
            return None

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        payload_dict = cast(dict[str, object], payload)
        name_raw = payload_dict.get("nameWithOwner")
        if isinstance(name_raw, str):
            self._repo_cache = name_raw
            return name_raw
        return None

    def _request_json(self, args: Sequence[str], *, context: str) -> object:
        result = self._runner.run(args, cwd=self._cwd, allow_error=True)
        if result.code != 0:
            message = result.stderr or result.stdout or "unknown error"
            raise RuntimeError(f"{context} failed ({result.code}): {message.strip()}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"{context} returned invalid JSON") from exc

    def classify_entity(self, number: str) -> str | None:
        """Return "issue" or "pull" if determinable, otherwise None."""

        self.ensure_available()
        repo = self._repo_name()
        if not repo:
            return None

        api_path = f"repos/{repo}/issues/{number}"
        try:
            payload = self._request_json(
                [self._gh_path or "gh", "api", api_path], context="Classify entity"
            )
        except RuntimeError:
            return None

        if isinstance(payload, dict) and "pull_request" in payload:
            return "pull"
        return "issue"

    def closing_issues(self, pr_number: str | None = None) -> list[str]:
        self.ensure_available()
        repo = self._repo_name()
        if not repo:
            return []

        args = [
            self._gh_path or "gh",
            "pr",
            "view",
            "--json",
            "closingIssuesReferences,number",
        ]
        if pr_number:
            args.insert(3, pr_number)

        try:
            payload = self._request_json(args, context="Fetch closing issues")
        except RuntimeError:
            return []

        if not isinstance(payload, dict):
            return []

        payload_dict = cast(dict[str, object], payload)
        issues_raw: object | None = payload_dict.get("closingIssuesReferences")
        numbers: list[str] = []
        if isinstance(issues_raw, list):
            issues_list = cast(list[object], issues_raw)
            for entry in issues_list:
                if not isinstance(entry, dict):
                    continue

                entry_dict = cast(dict[str, object], entry)
                number = entry_dict.get("number")
                if isinstance(number, int):
                    numbers.append(f"#{number}")

        return sorted(set(numbers))

    def issue_details(self, number: str) -> IssueDetails:
        self.ensure_available()
        repo = self._repo_name()
        if not repo:
            raise RuntimeError("Unable to resolve repository for issue lookup.")

        payload = self._request_json(
            [self._gh_path or "gh", "api", f"repos/{repo}/issues/{number}"],
            context="Fetch issue",
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected issue payload format.")

        payload_dict = cast(dict[str, object], payload)

        title_raw = payload_dict.get("title")
        title = title_raw if isinstance(title_raw, str) else ""

        body_raw = payload_dict.get("body")
        body = body_raw if isinstance(body_raw, str) else ""

        comments_url_raw = payload_dict.get("comments_url")
        comments_url = comments_url_raw if isinstance(comments_url_raw, str) else None

        comments: list[str] = []
        if comments_url:
            comment_payload = self._request_json(
                [self._gh_path or "gh", "api", f"{comments_url}?per_page=50"],
                context="Fetch issue comments",
            )
            if isinstance(comment_payload, list):
                comment_entries = cast(list[object], comment_payload)
                for entry in comment_entries:
                    if not isinstance(entry, dict):
                        continue
                    entry_dict = cast(dict[str, object], entry)
                    user_raw = entry_dict.get("user")
                    user_dict = (
                        cast(dict[str, object], user_raw)
                        if isinstance(user_raw, dict)
                        else {}
                    )
                    login_raw = user_dict.get("login")
                    login = login_raw if isinstance(login_raw, str) else ""
                    entry_body_raw = entry_dict.get("body")
                    entry_body = (
                        entry_body_raw if isinstance(entry_body_raw, str) else ""
                    )
                    comments.append(f"{login}: {entry_body}".strip())

        story_path = find_user_story_link(body)
        story_content = None
        if story_path:
            story_content = self.fetch_repo_file(story_path)

        return IssueDetails(
            number=f"#{number}",
            title=title or "(no title)",
            body=body or "(no body)",
            comments=comments,
            user_story_path=story_path,
            user_story_content=story_content,
        )

    def pr_details(self, number: str) -> PullRequestDetails:
        self.ensure_available()
        repo = self._repo_name()
        if not repo:
            raise RuntimeError("Unable to resolve repository for PR lookup.")

        payload = self._request_json(
            [self._gh_path or "gh", "api", f"repos/{repo}/pulls/{number}"],
            context="Fetch pull request",
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected pull request payload format.")

        payload_dict = cast(dict[str, object], payload)

        title_raw = payload_dict.get("title")
        title = title_raw if isinstance(title_raw, str) else ""

        body_raw = payload_dict.get("body")
        body = body_raw if isinstance(body_raw, str) else ""
        closing = self.closing_issues(number)

        return PullRequestDetails(
            number=f"#{number}",
            title=title or "(no title)",
            body=body or "(no body)",
            closing_issues=closing,
        )

    def current_pr(self) -> PullRequestDetails | None:
        if not self._available:
            return None

        args = [
            self._gh_path or "gh",
            "pr",
            "view",
            "--json",
            "number,title,body,closingIssuesReferences",
        ]
        result = self._runner.run(args, cwd=self._cwd, allow_error=True)
        if result.code != 0 or not result.stdout:
            return None

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        payload_dict = cast(dict[str, object], payload)

        closing_raw = payload_dict.get("closingIssuesReferences")
        closing: list[str] = []
        if isinstance(closing_raw, list):
            closing_entries = cast(list[object], closing_raw)
            for entry in closing_entries:
                if not isinstance(entry, dict):
                    continue
                entry_dict = cast(dict[str, object], entry)
                number = entry_dict.get("number")
                if isinstance(number, int):
                    closing.append(f"#{number}")

        title_raw = payload_dict.get("title")
        title = title_raw if isinstance(title_raw, str) else ""
        body_raw = payload_dict.get("body")
        body = body_raw if isinstance(body_raw, str) else ""
        number_raw = payload_dict.get("number")
        number = number_raw if isinstance(number_raw, int) else None
        if number is None:
            return None

        return PullRequestDetails(
            number=f"#{number}",
            title=title or "(no title)",
            body=body or "(no body)",
            closing_issues=sorted(set(closing)),
        )

    def fetch_repo_file(self, path: str) -> str | None:
        self.ensure_available()
        repo = self._repo_name()
        if not repo:
            return None

        api_path = f"repos/{repo}/contents/{path.lstrip('/')}"
        try:
            payload = self._request_json(
                [self._gh_path or "gh", "api", api_path], context="Fetch file"
            )
        except RuntimeError:
            return None

        if not isinstance(payload, dict):
            return None

        payload_dict = cast(dict[str, object], payload)

        content_raw = payload_dict.get("content")
        if not isinstance(content_raw, str):
            return None

        try:
            decoded = base64.b64decode(content_raw)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return None
        return decoded.decode("utf-8", errors="replace")

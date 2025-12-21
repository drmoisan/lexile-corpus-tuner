"""Unit tests for scripts.dev_tools.pr_context.git module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.dev_tools.pr_context.git import (
    CommandRunner,
    GitClient,
    SubprocessRunner,
)
from scripts.dev_tools.pr_context.models import CommandResult


class TestSubprocessRunner:
    """Test SubprocessRunner command execution."""

    def test_run_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """SubprocessRunner returns CommandResult on success."""
        mock_completed = Mock()
        mock_completed.stdout = "output\n"
        mock_completed.stderr = "warning\n"
        mock_completed.returncode = 0

        mock_run = Mock(return_value=mock_completed)
        monkeypatch.setattr("subprocess.run", mock_run)

        runner = SubprocessRunner()
        result = runner.run(["echo", "test"], cwd=tmp_path)

        assert result.stdout == "output"
        assert result.stderr == "warning"
        assert result.code == 0
        mock_run.assert_called_once()

    def test_run_failure_with_allow_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SubprocessRunner returns error result when allow_error=True."""
        mock_completed = Mock()
        mock_completed.stdout = ""
        mock_completed.stderr = "error message\n"
        mock_completed.returncode = 1

        mock_run = Mock(return_value=mock_completed)
        monkeypatch.setattr("subprocess.run", mock_run)

        runner = SubprocessRunner()
        result = runner.run(["false"], allow_error=True)

        assert result.code == 1
        assert result.stderr == "error message"

    def test_run_failure_raises_without_allow_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SubprocessRunner raises RuntimeError when command fails."""
        mock_completed = Mock()
        mock_completed.stdout = "out"
        mock_completed.stderr = "err"
        mock_completed.returncode = 128

        mock_run = Mock(return_value=mock_completed)
        monkeypatch.setattr("subprocess.run", mock_run)

        runner = SubprocessRunner()
        with pytest.raises(RuntimeError, match=r"git status failed \(128\)"):
            runner.run(["git", "status"], allow_error=False)

    def test_run_strips_trailing_newlines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SubprocessRunner strips trailing newlines from stdout/stderr."""
        mock_completed = Mock()
        mock_completed.stdout = "line1\nline2\n\n"
        mock_completed.stderr = "warn\n"
        mock_completed.returncode = 0

        mock_run = Mock(return_value=mock_completed)
        monkeypatch.setattr("subprocess.run", mock_run)

        runner = SubprocessRunner()
        result = runner.run(["cmd"])

        assert result.stdout == "line1\nline2"
        assert result.stderr == "warn"

    def test_run_handles_none_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SubprocessRunner handles None stdout/stderr."""
        mock_completed = Mock()
        mock_completed.stdout = None
        mock_completed.stderr = None
        mock_completed.returncode = 0

        mock_run = Mock(return_value=mock_completed)
        monkeypatch.setattr("subprocess.run", mock_run)

        runner = SubprocessRunner()
        result = runner.run(["cmd"])

        assert result.stdout == ""
        assert result.stderr == ""


class TestGitClient:
    """Test GitClient git command wrappers."""

    @pytest.fixture
    def mock_runner(self) -> Mock:
        """Mock CommandRunner for testing."""
        return Mock(spec=CommandRunner)

    @pytest.fixture
    def git_client(self, mock_runner: Mock, tmp_path: Path) -> GitClient:
        """GitClient instance with mock runner."""
        return GitClient(mock_runner, tmp_path)

    def test_run_delegates_to_runner(
        self, git_client: GitClient, mock_runner: Mock
    ) -> None:
        """GitClient.run delegates to runner with git prefix."""
        mock_runner.run.return_value = CommandResult("ok", "", 0)

        result = git_client.run(["status", "-s"])

        mock_runner.run.assert_called_once_with(
            ["git", "status", "-s"], cwd=git_client.cwd, allow_error=False
        )
        assert result.stdout == "ok"

    def test_run_passes_allow_error(
        self, git_client: GitClient, mock_runner: Mock
    ) -> None:
        """GitClient.run passes allow_error flag."""
        mock_runner.run.return_value = CommandResult("", "err", 1)

        result = git_client.run(["diff"], allow_error=True)

        mock_runner.run.assert_called_once_with(
            ["git", "diff"], cwd=git_client.cwd, allow_error=True
        )
        assert result.code == 1

    def test_resolve_root_when_git_dir_exists(
        self, git_client: GitClient, mock_runner: Mock, tmp_path: Path
    ) -> None:
        """GitClient.resolve_root returns cwd when .git exists."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        root = git_client.resolve_root()

        assert root == tmp_path
        mock_runner.run.assert_not_called()

    def test_resolve_root_when_git_dir_missing(
        self, git_client: GitClient, mock_runner: Mock, tmp_path: Path
    ) -> None:
        """GitClient.resolve_root calls rev-parse when .git missing."""
        mock_runner.run.return_value = CommandResult("/repo/root", "", 0)

        root = git_client.resolve_root()

        mock_runner.run.assert_called_once_with(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=git_client.cwd,
            allow_error=False,
        )
        assert root == Path("/repo/root")

    def test_rev_parse(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.rev_parse returns commit SHA."""
        mock_runner.run.return_value = CommandResult("abc123def", "", 0)

        sha = git_client.rev_parse("HEAD")

        mock_runner.run.assert_called_once_with(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=git_client.cwd,
            allow_error=False,
        )
        assert sha == "abc123def"

    def test_remote_verbose(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.remote_verbose returns remote list."""
        mock_runner.run.return_value = CommandResult("origin\turl (fetch)", "", 0)

        remotes = git_client.remote_verbose()

        mock_runner.run.assert_called_once_with(
            ["git", "remote", "-v"], cwd=git_client.cwd, allow_error=False
        )
        assert remotes == "origin\turl (fetch)"

    def test_branch_name(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.branch_name returns current branch."""
        mock_runner.run.return_value = CommandResult("feature/test", "", 0)

        branch = git_client.branch_name()

        mock_runner.run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_client.cwd,
            allow_error=False,
        )
        assert branch == "feature/test"

    def test_upstream(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.upstream returns upstream branch."""
        mock_runner.run.return_value = CommandResult("origin/main", "", 0)

        upstream = git_client.upstream()

        mock_runner.run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=git_client.cwd,
            allow_error=True,
        )
        assert upstream == "origin/main"

    def test_upstream_when_none(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.upstream returns empty string when no upstream."""
        mock_runner.run.return_value = CommandResult("", "no upstream", 128)

        upstream = git_client.upstream()

        assert upstream == ""

    def test_status_short(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.status_short returns short status."""
        mock_runner.run.return_value = CommandResult("## main\n M file.py", "", 0)

        status = git_client.status_short()

        mock_runner.run.assert_called_once_with(
            ["git", "status", "-sb"], cwd=git_client.cwd, allow_error=False
        )
        assert status == "## main\n M file.py"

    def test_untracked(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.untracked returns untracked files."""
        mock_runner.run.return_value = CommandResult("new.txt\nother.py", "", 0)

        files = git_client.untracked()

        mock_runner.run.assert_called_once_with(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=git_client.cwd,
            allow_error=False,
        )
        assert files == "new.txt\nother.py"

    def test_diff_name_status_staged(
        self, git_client: GitClient, mock_runner: Mock
    ) -> None:
        """GitClient.diff_name_status with staged=True."""
        mock_runner.run.return_value = CommandResult("M\tfile.py", "", 0)

        diff = git_client.diff_name_status(staged=True)

        mock_runner.run.assert_called_once_with(
            ["git", "diff", "--cached", "--name-status"],
            cwd=git_client.cwd,
            allow_error=True,
        )
        assert diff == "M\tfile.py"

    def test_diff_name_status_unstaged(
        self, git_client: GitClient, mock_runner: Mock
    ) -> None:
        """GitClient.diff_name_status with staged=False."""
        mock_runner.run.return_value = CommandResult("A\tnew.py", "", 0)

        diff = git_client.diff_name_status(staged=False)

        mock_runner.run.assert_called_once_with(
            ["git", "diff", "--name-status"], cwd=git_client.cwd, allow_error=True
        )
        assert diff == "A\tnew.py"

    def test_diff_patch_staged(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.diff_patch with staged=True."""
        mock_runner.run.return_value = CommandResult("diff --git", "", 0)

        patch = git_client.diff_patch(staged=True)

        mock_runner.run.assert_called_once_with(
            ["git", "diff", "--cached"], cwd=git_client.cwd, allow_error=True
        )
        assert patch == "diff --git"

    def test_diff_patch_unstaged(
        self, git_client: GitClient, mock_runner: Mock
    ) -> None:
        """GitClient.diff_patch with staged=False."""
        mock_runner.run.return_value = CommandResult("diff content", "", 0)

        patch = git_client.diff_patch(staged=False)

        mock_runner.run.assert_called_once_with(
            ["git", "diff"], cwd=git_client.cwd, allow_error=True
        )
        assert patch == "diff content"

    def test_merge_base(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.merge_base returns common ancestor."""
        mock_runner.run.return_value = CommandResult("abc123", "", 0)

        base = git_client.merge_base("main", "feature")

        mock_runner.run.assert_called_once_with(
            ["git", "merge-base", "main", "feature"],
            cwd=git_client.cwd,
            allow_error=False,
        )
        assert base == "abc123"

    def test_log(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.log returns formatted log."""
        mock_runner.run.return_value = CommandResult("commit1\ncommit2", "", 0)

        log = git_client.log("--oneline", "main..feature")

        mock_runner.run.assert_called_once_with(
            ["git", "log", "--date=short", "--oneline", "main..feature"],
            cwd=git_client.cwd,
            allow_error=True,
        )
        assert log == "commit1\ncommit2"

    def test_diff_range(self, git_client: GitClient, mock_runner: Mock) -> None:
        """GitClient.diff_range returns diff for range."""
        mock_runner.run.return_value = CommandResult("diff output", "", 0)

        diff = git_client.diff_range(["--stat", "main", "feature", "--", "file.py"])

        mock_runner.run.assert_called_once_with(
            ["git", "diff", "--stat", "main", "feature", "--", "file.py"],
            cwd=git_client.cwd,
            allow_error=True,
        )
        assert diff == "diff output"

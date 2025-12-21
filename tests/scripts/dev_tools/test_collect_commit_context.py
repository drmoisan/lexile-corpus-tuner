"""Tests for collect_commit_context module."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.dev_tools.collect_commit_context import (
    collect_commit_context,
    main,
    run_git,
)


class TestRunGit:
    """Tests for run_git function."""

    def test_successful_command_returns_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that successful git command returns stripped stdout."""
        mock_result = subprocess.CompletedProcess(
            args=["git", "status"],
            returncode=0,
            stdout="  some output  \n",
            stderr="",
        )

        def mock_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return mock_result

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = run_git(["status"])
        assert result == "some output"

    def test_failed_command_raises_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that failed command raises CalledProcessError when allow_error=False."""

        def mock_run(*args: object, **kwargs: object) -> None:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["git", "status"],
                output="",
                stderr="fatal: not a git repository",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        with pytest.raises(subprocess.CalledProcessError):
            run_git(["status"])

    def test_failed_command_returns_empty_when_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that failed git command returns empty string when allow_error=True."""

        def mock_run(*args: object, **kwargs: object) -> None:
            exc = subprocess.CalledProcessError(
                returncode=1,
                cmd=["git", "status"],
                output="",
                stderr="fatal: not a git repository",
            )
            exc.stdout = ""
            raise exc

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = run_git(["status"], allow_error=True)
        assert result == ""

    def test_failed_command_returns_stdout_when_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that failed command returns stdout when allow_error=True."""

        def mock_run(*args: object, **kwargs: object) -> None:
            exc = subprocess.CalledProcessError(
                returncode=1,
                cmd=["git", "status"],
                output="partial output",
                stderr="error message",
            )
            exc.stdout = "  partial output  \n"
            raise exc

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = run_git(["status"], allow_error=True)
        assert result == "partial output"

    def test_subprocess_run_called_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that subprocess.run is called with the correct arguments."""
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def mock_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(
                args=["git", "status"],
                returncode=0,
                stdout="output",
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        run_git(["status", "-sb"])

        assert len(calls) == 1
        args_tuple, kwargs_dict = calls[0]
        assert args_tuple == (["git", "status", "-sb"],)
        assert kwargs_dict["capture_output"] is True
        assert kwargs_dict["text"] is True
        assert kwargs_dict["check"] is True


class TestCollectCommitContext:
    """Tests for collect_commit_context function."""

    def test_creates_output_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the function creates the output file."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            return "mock output"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "test_output.txt"
        collect_commit_context(output_file)

        assert output_file.exists()

    def test_creates_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the function creates parent directories if they don't exist."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            return "mock output"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "nested" / "dir" / "output.txt"
        collect_commit_context(output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_output_contains_expected_sections(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that output file contains all expected section headers."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "remote" in args:
                return "origin\thttps://github.com/user/repo.git (fetch)"
            if "rev-parse" in args and "--abbrev-ref" in args and "HEAD" in args:
                return "main"
            if "rev-parse" in args and "@{u}" in args:
                return "origin/main"
            if "status" in args:
                return "## main...origin/main"
            if "diff" in args and "--cached" in args and "--name-status" in args:
                return "M\tfile.py"
            if "diff" in args and "--cached" in args:
                return "diff --git a/file.py b/file.py"
            if "diff" in args and "--name-status" in args:
                return ""
            if "diff" in args and "HEAD" in args and "--stat" in args:
                return "1 file changed"
            if "diff" in args and "HEAD" in args and "--name-only" in args:
                return "file.py"
            if "ls-files" in args:
                return ""
            if "log" in args:
                return (
                    "abc123\nAuthor <author@example.com>\nMon Dec 18 2023\n"
                    "Committer <committer@example.com>\nMon Dec 18 2023\n"
                    "feat: add feature\n\ndetailed body"
                )
            return ""

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")

        # Check for all expected section headers
        assert "===== Repository remotes =====" in content
        assert "===== Current branch =====" in content
        assert "===== Upstream =====" in content
        assert "===== Status (short) =====" in content
        assert "===== Staged files (name-status) =====" in content
        assert "===== Staged diff =====" in content
        assert "===== Unstaged files (name-status) =====" in content
        assert "===== Unstaged diff =====" in content
        assert "===== Untracked files =====" in content
        assert "===== Diff stat (staged + unstaged) =====" in content
        assert "===== Changed Python files =====" in content
        assert "===== Last commit (header only) =====" in content
        assert "===== Change intent (edit below) =====" in content

    def test_handles_no_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing upstream is handled gracefully."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "@{u}" in args:
                return ""  # No upstream
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no upstream)" in content

    def test_handles_no_staged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no staged changes is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "diff" in args and "--cached" in args:
                return ""
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no staged changes)" in content

    def test_handles_no_unstaged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no unstaged changes is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "diff" in args and "--cached" not in args and "--name-status" in args:
                return ""
            if "diff" in args and "--cached" not in args and "HEAD" not in args:
                return ""
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no unstaged changes)" in content

    def test_handles_no_untracked_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no untracked files is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "ls-files" in args:
                return ""
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no untracked files)" in content

    def test_handles_no_changes_in_diff_stat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no changes in diff stat is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "diff" in args and "HEAD" in args and "--stat" in args:
                return ""
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no changes)" in content

    def test_filters_python_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Python files are filtered correctly from changed files."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "diff" in args and "HEAD" in args and "--name-only" in args:
                return "file1.py\nfile2.txt\nfile3.py\nREADME.md"
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "file1.py" in content
        assert "file3.py" in content
        assert "file2.txt" not in content
        assert "README.md" not in content

    def test_handles_no_python_files_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no Python files changed is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "diff" in args and "HEAD" in args and "--name-only" in args:
                return "file.txt\nREADME.md"
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no Python files changed)" in content

    def test_handles_no_previous_commits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that no previous commits is indicated correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "log" in args:
                return ""
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "(no previous commits)" in content

    def test_formats_last_commit_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that last commit is formatted with all fields correctly."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            if "log" in args:
                return (
                    "abc123def456\n"
                    "John Doe <john@example.com>\n"
                    "Mon Dec 18 10:30:00 2023 -0500\n"
                    "Jane Committer <jane@example.com>\n"
                    "Mon Dec 18 10:35:00 2023 -0500\n"
                    "feat: add new feature\n"
                    "\n"
                    "This is a detailed description\n"
                    "spanning multiple lines"
                )
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "commit abc123def456" in content
        assert "Author:     John Doe <john@example.com>" in content
        assert "AuthorDate: Mon Dec 18 10:30:00 2023 -0500" in content
        assert "Commit:     Jane Committer <jane@example.com>" in content
        assert "CommitDate: Mon Dec 18 10:35:00 2023 -0500" in content
        assert "    feat: add new feature" in content
        assert "    This is a detailed description" in content
        assert "    spanning multiple lines" in content

    def test_prints_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that the output path is printed to stdout."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        collect_commit_context(output_file)

        captured = capsys.readouterr()
        assert str(output_file) in captured.out


class TestMain:
    """Tests for main function."""

    def test_successful_execution_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that successful execution returns exit code 0."""

        def mock_run_git(args: list[str], allow_error: bool = False) -> str:
            return "mock"

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.run_git", mock_run_git
        )

        output_file = tmp_path / "output.txt"
        exit_code = main(["--output", str(output_file)])

        assert exit_code == 0
        assert output_file.exists()

    def test_uses_default_output_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that default output path is used when not specified."""
        calls: list[Path] = []

        def mock_collect(output_path: Path) -> None:
            calls.append(output_path)

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.collect_commit_context",
            mock_collect,
        )

        # Change cwd to tmp_path to avoid writing to real artifacts directory
        monkeypatch.chdir(tmp_path)

        main([])

        assert len(calls) == 1
        assert calls[0] == Path("artifacts/commit_context.txt")

    def test_accepts_custom_output_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that custom output path is used when specified."""
        calls: list[Path] = []

        def mock_collect(output_path: Path) -> None:
            calls.append(output_path)

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.collect_commit_context",
            mock_collect,
        )

        custom_path = tmp_path / "custom_output.txt"
        main(["--output", str(custom_path)])

        assert len(calls) == 1
        assert calls[0] == custom_path

    def test_accepts_short_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that -o short flag is accepted."""
        calls: list[Path] = []

        def mock_collect(output_path: Path) -> None:
            calls.append(output_path)

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.collect_commit_context",
            mock_collect,
        )

        custom_path = tmp_path / "custom_output.txt"
        main(["-o", str(custom_path)])

        assert len(calls) == 1
        assert calls[0] == custom_path

    def test_returns_one_on_subprocess_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that subprocess.CalledProcessError results in exit code 1."""

        def mock_collect(output_path: Path) -> None:
            raise subprocess.CalledProcessError(1, ["git", "status"])

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.collect_commit_context",
            mock_collect,
        )

        output_file = tmp_path / "output.txt"
        exit_code = main(["--output", str(output_file)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Git command failed" in captured.err

    def test_returns_one_on_general_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that general exceptions result in exit code 1."""

        def mock_collect(output_path: Path) -> None:
            raise ValueError("Test error")

        monkeypatch.setattr(
            "scripts.dev_tools.collect_commit_context.collect_commit_context",
            mock_collect,
        )

        output_file = tmp_path / "output.txt"
        exit_code = main(["--output", str(output_file)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "Test error" in captured.err

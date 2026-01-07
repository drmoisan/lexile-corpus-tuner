"""
Tests for atomic_executor.cli module.

Tests cover CLI argument parsing, workspace resolution, precondition checks,
clipboard operations, and main execution orchestration.
"""

# pyright: reportArgumentType=false

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from scripts.dev_tools.atomic_executor.cli import (
    copy_to_clipboard,
    ensure_clean_tree,
    parse_args,
    refuse_protected_branch,
    resolve_workspace,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


class TestParseArgs:
    """Tests for parse_args() function."""

    def test_parse_execute_subcommand_with_path(self) -> None:
        """parse_args() parses execute subcommand with required path."""
        args = parse_args(["execute", "feature-folder"])
        assert args.cmd == "execute"
        assert args.path == "feature-folder"
        assert args.workspace is None
        assert args.feature is None
        assert args.prompt_template == ".github/prompts/execute-atomic-plan.prompt.md"
        assert args.start is None
        assert args.max_fix_attempts == 2
        assert args.print_prompt is False
        assert args.copy_prompt is False

    def test_parse_resume_subcommand(self) -> None:
        """parse_args() parses resume subcommand."""
        args = parse_args(["resume", "feature-folder"])
        assert args.cmd == "resume"
        assert args.path == "feature-folder"

    def test_parse_with_all_optional_args(self) -> None:
        """parse_args() parses all optional arguments."""
        args = parse_args(
            [
                "execute",
                "my-feature",
                "--workspace",
                "/tmp/repo",  # noqa: S108
                "--feature",
                "my-feat",
                "--prompt-template",
                "custom.md",
                "--start",
                "P2-T5",
                "--max-fix-attempts",
                "5",
                "--print-prompt",
            ]
        )
        assert args.workspace == "/tmp/repo"  # noqa: S108
        assert args.feature == "my-feat"
        assert args.prompt_template == "custom.md"
        assert args.start == "P2-T5"
        assert args.max_fix_attempts == 5
        assert args.print_prompt is True

    def test_parse_copy_prompt_flag(self) -> None:
        """parse_args() parses --copy-prompt flag."""
        args = parse_args(["execute", "feature", "--copy-prompt"])
        assert args.copy_prompt is True

    def test_parse_raises_for_missing_subcommand(self) -> None:
        """parse_args() raises SystemExit when subcommand missing."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_raises_for_missing_path(self) -> None:
        """parse_args() raises SystemExit when path argument missing."""
        with pytest.raises(SystemExit):
            parse_args(["execute"])


class TestResolveWorkspace:
    """Tests for resolve_workspace() function."""

    def test_resolve_uses_explicit_workspace(self, tmp_path: Path) -> None:
        """resolve_workspace() uses explicit workspace argument."""
        result = resolve_workspace(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_resolve_infers_from_file_location(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve_workspace() infers from __file__ location when no arg."""
        # Mock __file__ to be at repo/scripts/dev_tools/atomic_executor/cli.py
        fake_file = Path("/fake/repo/scripts/dev_tools/atomic_executor/cli.py")
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli.__file__", str(fake_file)
        )

        result = resolve_workspace(None)
        # Should go up 3 levels: atomic_executor -> dev_tools -> scripts -> repo
        # resolve() normalizes path on Windows
        assert result.resolve() == Path("/fake/repo").resolve()


class TestEnsureCleanTree:
    """Tests for ensure_clean_tree() function."""

    def test_ensure_clean_tree_passes_for_clean_tree(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """ensure_clean_tree() passes when tree is clean."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        # Should not raise
        ensure_clean_tree(tmp_path)

    def test_ensure_clean_tree_raises_for_dirty_tree(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """ensure_clean_tree() raises RuntimeError when tree has changes."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = " M modified_file.py\n"
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        with pytest.raises(RuntimeError, match="Working tree is not clean"):
            ensure_clean_tree(tmp_path)


class TestRefuseProtectedBranch:
    """Tests for refuse_protected_branch() function."""

    def test_refuse_raises_for_main_branch(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """refuse_protected_branch() raises for 'main' branch."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "main\n"
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        with pytest.raises(RuntimeError, match="protected branch"):
            refuse_protected_branch(tmp_path)

    def test_refuse_raises_for_master_branch(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """refuse_protected_branch() raises for 'master' branch."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "master\n"
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        with pytest.raises(RuntimeError, match="protected branch"):
            refuse_protected_branch(tmp_path)

    def test_refuse_raises_for_development_branch(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """refuse_protected_branch() raises for 'development' branch."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "development\n"
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        with pytest.raises(RuntimeError, match="protected branch"):
            refuse_protected_branch(tmp_path)

    def test_refuse_passes_for_feature_branch(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """refuse_protected_branch() passes for non-protected branch."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/my-feature\n"
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        # Should not raise
        refuse_protected_branch(tmp_path)

    def test_refuse_handles_git_error(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """refuse_protected_branch() handles git errors gracefully."""

        def mock_run(*args: object, **kwargs: object) -> Mock:
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr("subprocess.run", mock_run)

        # Should not raise when git fails (returns None, not in
        # PROTECTED_BRANCHES)
        refuse_protected_branch(tmp_path)


class TestCopyToClipboard:
    """Tests for copy_to_clipboard() function."""

    def test_copy_uses_pyperclip_when_available(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """copy_to_clipboard() uses pyperclip when available."""
        copied_text = None

        def mock_copy(text: str) -> None:
            nonlocal copied_text
            copied_text = text

        # Create a mock pyperclip module
        mock_pyperclip = Mock()
        mock_pyperclip.copy = mock_copy

        # Mock the import at import time
        import builtins

        original_import = builtins.__import__

        def custom_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def,misc]
            if name == "pyperclip":
                return mock_pyperclip
            return original_import(name, *args, **kwargs)  # type: ignore[call-arg]

        monkeypatch.setattr("builtins.__import__", custom_import)

        result = copy_to_clipboard("test text")

        assert result is True
        assert copied_text == "test text"

    def test_copy_falls_back_to_command_when_pyperclip_unavailable(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """copy_to_clipboard() falls back to commands when pyperclip fails."""
        # Mock pyperclip import to fail
        import builtins

        original_import = builtins.__import__

        def custom_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if name == "pyperclip":
                raise ImportError("pyperclip not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)

        captured_input = None

        def mock_which(cmd: str) -> str | None:
            if cmd == "clip":
                return "clip.exe"
            return None

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal captured_input
            captured_input = kwargs.get("input")
            result = Mock()
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("subprocess.run", mock_run)

        result = copy_to_clipboard("fallback text")

        assert result is True
        assert captured_input == "fallback text"

    def test_copy_returns_false_when_all_methods_fail(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """copy_to_clipboard() returns False when all methods fail."""
        # Mock pyperclip import to fail
        import builtins

        original_import = builtins.__import__

        def custom_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if name == "pyperclip":
                raise ImportError("pyperclip not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)

        # Mock which to return None for all commands
        monkeypatch.setattr("shutil.which", lambda x: None)  # type: ignore[arg-type,misc]

        result = copy_to_clipboard("text")

        assert result is False

    def test_copy_tries_multiple_fallback_commands(
        self, monkeypatch: "MonkeyPatch"
    ) -> None:
        """copy_to_clipboard() tries multiple fallback commands."""
        # Mock pyperclip import to fail
        import builtins

        original_import = builtins.__import__

        def custom_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if name == "pyperclip":
                raise ImportError("pyperclip not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)

        attempted_commands: list[str] = []

        def mock_which(cmd: str) -> str | None:
            # Make xclip available
            if cmd == "xclip":
                return "/usr/bin/xclip"
            return None

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            attempted_commands.append(argv[0])
            result = Mock()
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("subprocess.run", mock_run)

        result = copy_to_clipboard("text")

        assert result is True
        assert "/usr/bin/xclip" in attempted_commands


class TestMainEdgeCases:
    """Edge case tests for main execution flow."""

    def test_main_exits_early_with_print_prompt(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() exits early with --print-prompt without running copilot."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup minimal feature folder
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        (feature_dir / "plan.md").write_text(
            "# Phase 0\n- [ ] [P0-T1] Task 1\n\n"
            "# Phase 2 (QA/Toolchain)\n"
            "- [ ] [P2-T1] Black\n"
            "- [ ] [P2-T2] Ruff\n"
            "- [ ] [P2-T3] Pyright\n"
            "- [ ] [P2-T4] Pytest",
            encoding="utf-8",
        )
        (feature_dir / "spec.md").write_text("# Spec\n", encoding="utf-8")

        template_dir = tmp_path / ".github" / "prompts"
        template_dir.mkdir(parents=True)
        (template_dir / "execute-atomic-plan.prompt.md").write_text(
            "TEMPLATE\n", encoding="utf-8"
        )

        # Mock all subprocess calls
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        # Mock copilot to not be found (shouldn't be called)
        monkeypatch.setattr("shutil.which", lambda x: None if x == "copilot" else "git")  # type: ignore[arg-type,misc]

        exit_code = main(
            [
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
                "--print-prompt",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "TEMPLATE" in captured.out
        assert "CURRENT TASK" in captured.out

    def test_main_exits_early_with_copy_prompt(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() exits early with --copy-prompt without running copilot."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup minimal feature folder
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        (feature_dir / "plan.md").write_text(
            "# Phase 0\n- [ ] [P0-T1] Task 1\n\n"
            "# Phase 2 (QA/Toolchain)\n"
            "- [ ] [P2-T1] Black\n"
            "- [ ] [P2-T2] Ruff\n"
            "- [ ] [P2-T3] Pyright\n"
            "- [ ] [P2-T4] Pytest",
            encoding="utf-8",
        )
        (feature_dir / "spec.md").write_text("# Spec\n", encoding="utf-8")

        template_dir = tmp_path / ".github" / "prompts"
        template_dir.mkdir(parents=True)
        (template_dir / "execute-atomic-plan.prompt.md").write_text(
            "TEMPLATE\n", encoding="utf-8"
        )

        # Mock all subprocess calls
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        # Mock clipboard to succeed
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli.copy_to_clipboard", lambda x: True  # type: ignore[arg-type,misc]
        )

        exit_code = main(
            [
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
                "--copy-prompt",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "copied to clipboard" in captured.err.lower()

    def test_main_returns_error_for_missing_plan(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() returns error code when plan.md missing."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup feature folder without plan.md
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        (feature_dir / "spec.md").write_text("# Spec\n", encoding="utf-8")

        # Mock git to be clean
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        exit_code = main(
            [
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
            ]
        )

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Missing required plan.md" in captured.err

    def test_main_returns_zero_when_plan_already_complete(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() returns 0 when all tasks already checked."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup feature folder with all tasks checked
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        (feature_dir / "plan.md").write_text(
            "# Phase 0\n- [x] [P0-T1] Task 1\n\n"
            "# Phase 2 (QA/Toolchain)\n"
            "- [x] [P2-T1] Black\n"
            "- [x] [P2-T2] Ruff\n"
            "- [x] [P2-T3] Pyright\n"
            "- [x] [P2-T4] Pytest",
            encoding="utf-8",
        )
        (feature_dir / "spec.md").write_text("# Spec\n", encoding="utf-8")

        template_dir = tmp_path / ".github" / "prompts"
        template_dir.mkdir(parents=True)
        (template_dir / "execute-atomic-plan.prompt.md").write_text(
            "TEMPLATE\n", encoding="utf-8"
        )

        # Mock git to be clean
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        exit_code = main(
            [
                "resume",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "already complete" in captured.out.lower()

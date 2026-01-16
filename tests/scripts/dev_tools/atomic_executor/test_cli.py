"""
Tests for atomic_executor.cli module.

Tests cover CLI argument parsing, workspace resolution, precondition checks,
clipboard operations, and main execution orchestration.
"""

# pyright: reportArgumentType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

import os
import subprocess
import sys
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
        assert args.prompt_template == ".github/prompts/execute-plan-template.md"
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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/git")

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
        # Use Windows platform to deterministically pick 'clip' which is mocked below
        monkeypatch.setattr(sys, "platform", "win32")

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
        """copy_to_clipboard() tries multiple fallback commands on Linux."""
        # Mock pyperclip import to fail
        import builtins
        import sys

        original_import = builtins.__import__

        def custom_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if name == "pyperclip":
                raise ImportError("pyperclip not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)
        # Mock platform to Linux so it uses xclip/wl-copy candidates
        monkeypatch.setattr(sys, "platform", "linux")

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
        (template_dir / "execute-plan-template.md").write_text(
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
        (template_dir / "execute-plan-template.md").write_text(
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
        assert "Missing required plan file" in captured.err

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
        (template_dir / "execute-plan-template.md").write_text(
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

    def test_main_returns_error_for_missing_template(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() returns error code when prompt template missing."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup feature folder with plan.md
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

        # Don't create template file

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
        assert "Prompt template not found" in captured.err

    def test_main_with_copy_prompt_fallback_when_clipboard_fails(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() prints prompt when --copy-prompt fails."""
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
        (template_dir / "execute-plan-template.md").write_text(
            "TEMPLATE\n", encoding="utf-8"
        )

        # Mock all subprocess calls
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        monkeypatch.setattr("subprocess.run", mock_run)

        # Mock clipboard to fail
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli.copy_to_clipboard",
            lambda x: False,  # type: ignore[arg-type,misc]
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
        assert "Clipboard copy not available" in captured.err
        assert "TEMPLATE" in captured.out

    def test_main_execute_with_start_flag(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() executes with --start flag to begin at specific task."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup feature folder with multiple tasks
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        (feature_dir / "plan.md").write_text(
            "# Phase 0\n"
            "- [x] [P0-T1] First task\n"
            "- [ ] [P0-T2] Second task\n\n"
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
        (template_dir / "execute-plan-template.md").write_text(
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
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
                "--start",
                "P0-T2",
                "--print-prompt",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "P0-T2" in captured.out
        assert "Second task" in captured.out

    def test_main_execute_when_all_tasks_complete(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() exits when execute subcommand finds no unchecked tasks."""
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
        (template_dir / "execute-plan-template.md").write_text(
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
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "already complete" in captured.out.lower()

    def test_main_successful_execution_with_scoped_qc(
        self,
        tmp_path: Path,
        monkeypatch: "MonkeyPatch",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() successfully executes task with scoped QC."""
        from scripts.dev_tools.atomic_executor.cli import main

        # Setup feature folder
        feature_dir = tmp_path / "docs" / "features" / "active" / "my-feature"
        feature_dir.mkdir(parents=True)
        plan_file = feature_dir / "plan.md"
        plan_file.write_text(
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
        (template_dir / "execute-plan-template.md").write_text(
            "Task: {{task_id}}\n", encoding="utf-8"
        )

        # Setup fake copilot on PATH for run_copilot
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        copilot_exe = bin_dir / "copilot"
        copilot_exe.touch()
        copilot_exe.chmod(0o755)  # Make executable-ish
        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{str(bin_dir)}{os.pathsep}{path}")

        # Track subprocess calls
        subprocess_calls: list[list[str]] = []

        def mock_run(
            argv: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            subprocess_calls.append(argv)
            result = Mock()
            result.stdout = ""
            result.returncode = 0
            return result  # type: ignore[return-value]

        class MockStdout:
            def read(self, size: int = -1) -> bytes:
                return b""

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                subprocess_calls.append(argv)
                self.stdout = MockStdout()
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("subprocess.Popen", MockPopen)
        monkeypatch.setattr("shutil.which", lambda x: f"/usr/bin/{x}")

        # Mock QCRunner methods to succeed
        from scripts.dev_tools.atomic_executor.qc_runner import QCRunner

        monkeypatch.setattr(QCRunner, "run_scoped", lambda self: None)
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        exit_code = main(
            [
                "execute",
                str(feature_dir),
                "--workspace",
                str(tmp_path),
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "complete and gated" in captured.out.lower()

        # Verify copilot was invoked
        copilot_calls = [c for c in subprocess_calls if "copilot" in c[0]]
        assert len(copilot_calls) == 1


class TestRunCopilot:
    """Tests for run_copilot() function."""

    def test_run_copilot_raises_when_executable_not_found(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() raises FileNotFoundError when copilot not found."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # Set PATH to empty so no copilot executable can be found
        monkeypatch.setenv("PATH", "")

        log_file = tmp_path / "test.log"

        with pytest.raises(
            FileNotFoundError,
            match=r"Required executable not found on PATH: copilot\b",
        ):
            run_copilot(
                workspace=tmp_path,
                prompt_text="test prompt",
                log_file=log_file,
                task_id="P1-T1",
                preferred_model=None,
                run_id="2026-01-07_000000",
            )

    def test_run_copilot_rejects_vscode_shim(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() skips VS Code shim and finds no other copilot."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # Create shim directory structure that matches the detection pattern
        # Pattern: .../Code/User/globalStorage/github.copilot-chat/copilotCli/
        # Use nested dirs to ensure we hit the pattern matching logic
        shim_dir = (
            tmp_path
            / "Code"
            / "User"
            / "globalStorage"
            / "github.copilot-chat"
            / "copilotCli"
        )
        shim_dir.mkdir(parents=True)

        # Create executable shim
        # In cli.py, it checks specifically for copilot.exe, copilot.bat, copilot
        fake_shim = shim_dir / "copilot"
        fake_shim.touch()
        fake_shim.chmod(0o755)

        # Update PATH to point to this directory
        monkeypatch.setenv("PATH", str(shim_dir))

        log_file = tmp_path / "test.log"

        with pytest.raises(
            FileNotFoundError,
            match=r"Required executable not found on PATH: copilot\b",
        ):
            run_copilot(
                workspace=tmp_path,
                prompt_text="test prompt",
                log_file=log_file,
                task_id="P1-T1",
                preferred_model=None,
                run_id="2026-01-07_000000",
            )

    def test_run_copilot_rejects_vscode_shim_remote_paths(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() rejects the VS Code Remote/Devcontainer shim path."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # VS Code Remote (including devcontainers) stores its shim under a Linux
        # path, e.g. ~/.vscode-server/data/User/globalStorage/github.copilot-chat/
        # copilotCli/. If we accidentally execute this shim, it can block waiting
        # for interactive install/auth and appear as a hang.
        shim_dir = (
            tmp_path
            / ".vscode-server"
            / "data"
            / "User"
            / "globalStorage"
            / "github.copilot-chat"
            / "copilotCli"
        )
        shim_dir.mkdir(parents=True)

        fake_shim = shim_dir / "copilot"
        fake_shim.touch()
        fake_shim.chmod(0o755)

        # Ensure the shim is the ONLY PATH entry.
        monkeypatch.setenv("PATH", str(shim_dir))

        # Guard: if the implementation tries to execute the shim, fail the test.
        def _should_not_invoke_popen(*args: object, **kwargs: object) -> None:
            pytest.fail("run_copilot attempted to execute a VS Code shim")

        monkeypatch.setattr("subprocess.Popen", _should_not_invoke_popen)

        log_file = tmp_path / "test.log"

        with pytest.raises(
            FileNotFoundError,
            match=r"Required executable not found on PATH: copilot\b",
        ):
            run_copilot(
                workspace=tmp_path,
                prompt_text="test prompt",
                log_file=log_file,
                task_id="P1-T1",
                preferred_model=None,
                run_id="2026-01-07_000000",
            )

    def test_run_copilot_creates_log_directory(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() creates log directory if missing."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # Setup fake copilot on PATH
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        copilot_exe = bin_dir / "copilot"
        copilot_exe.touch()
        copilot_exe.chmod(0o755)
        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{str(bin_dir)}{os.pathsep}{path}")

        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        class MockStdout:
            def read(self, size: int = -1) -> bytes:
                return b""

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                self.stdout = MockStdout()
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", MockPopen)

        # Also mock run purely to avoid confusion, though unused by run_copilot directly
        def mock_run(*args: object, **kwargs: object) -> Mock:
            return Mock(returncode=0)

        monkeypatch.setattr("subprocess.run", mock_run)

        log_dir = tmp_path / "nested" / "log" / "dir"
        log_file = log_dir / "test.log"

        run_copilot(
            workspace=tmp_path,
            prompt_text="test prompt",
            log_file=log_file,
            task_id="P1-T1",
            preferred_model=None,
            run_id="2026-01-07_000000",
        )

        assert log_dir.exists()
        assert log_file.exists()

    def test_run_copilot_invokes_with_correct_arguments(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() invokes copilot with correct arguments."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # Create a fake copilot executable on PATH
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_copilot = fake_bin / "copilot.exe"
        fake_copilot.write_text("@echo fake copilot")
        monkeypatch.setenv("PATH", str(fake_bin))
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        captured_argv: list[str] = []
        captured_stdin: str | None = None
        captured_stdin_was_provided = False

        class MockStdout:
            def read(self, size: int = -1) -> bytes:
                return b""

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                nonlocal captured_stdin
                nonlocal captured_stdin_was_provided
                captured_argv.extend(argv)
                # Capture stdin content if provided
                stdin_arg = kwargs.get("stdin")
                captured_stdin_was_provided = (
                    "stdin" in kwargs and stdin_arg is not None
                )
                if stdin_arg and hasattr(stdin_arg, "read"):
                    # Type narrowing: we know it has a read method and returns bytes
                    raw_content = stdin_arg.read()  # type: ignore[union-attr]
                    # Explicitly cast to bytes for type checker
                    content_bytes: bytes = (
                        raw_content if isinstance(raw_content, bytes) else b""
                    )
                    captured_stdin = content_bytes.decode("utf-8")
                self.stdout = MockStdout()
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", MockPopen)

        log_file = tmp_path / "test.log"

        run_copilot(
            workspace=tmp_path,
            prompt_text="test prompt",
            log_file=log_file,
            task_id="P1-T1",
            preferred_model="gpt-5.1-codex-max",
            run_id="2026-01-07_000000",
        )

        expected_prompt_file = (
            log_file.parent / "prompts" / "prompt_2026-01-07_000000_P1-T1.md"
        )
        assert expected_prompt_file.exists()
        assert expected_prompt_file.read_text(encoding="utf-8") == "test prompt"

        assert captured_argv[0] == str(fake_copilot)
        assert "--model" in captured_argv
        assert "gpt-5.1-codex-max" in captured_argv
        assert "--session-path" not in captured_argv
        # Prompt must be passed via programmatic mode (-p/--prompt) with an @path
        # reference to the on-disk prompt file.
        assert "-p" in captured_argv or "--prompt" in captured_argv
        prompt_flag = "-p" if "-p" in captured_argv else "--prompt"
        prompt_idx = captured_argv.index(prompt_flag)
        prompt_value = captured_argv[prompt_idx + 1]
        assert f"@{expected_prompt_file}" in prompt_value

        # Prompt must NOT be provided via stdin.
        assert captured_stdin_was_provided is False
        assert captured_stdin is None
        assert "--share" in captured_argv
        assert "--allow-tool" in captured_argv
        assert "write" in captured_argv

        # Tool approvals required for headless QC must remain present.
        assert "shell(poetry)" in captured_argv
        assert "shell(git)" in captured_argv

    def test_run_copilot_permission_denied_fails_fast_with_actionable_error(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() raises promptly when Copilot reports a permission denial."""

        from scripts.dev_tools.atomic_executor.cli import run_copilot

        # Create a fake copilot executable on PATH.
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_copilot = fake_bin / "copilot.exe"
        fake_copilot.write_text("@echo fake copilot")
        monkeypatch.setenv("PATH", str(fake_bin))
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        captured_argv: list[str] = []

        permission_denied = (
            "Permission denied and could not request permission from user"
        )

        class MockStdout:
            def __init__(self) -> None:
                self._chunks = [permission_denied.encode("utf-8"), b""]

            def read(self, size: int = -1) -> bytes:
                return self._chunks.pop(0) if self._chunks else b""

            def read1(self, size: int = -1) -> bytes:
                return self.read(size)

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                captured_argv.extend(argv)
                self.stdout = MockStdout()
                self.returncode = 1

            def poll(self) -> int:
                return 1

            def wait(self, timeout: float | None = None) -> int:
                return 1

        monkeypatch.setattr("subprocess.Popen", MockPopen)

        log_file = tmp_path / "test.log"

        with pytest.raises(RuntimeError) as exc_info:
            run_copilot(
                workspace=tmp_path,
                prompt_text="test prompt",
                log_file=log_file,
                task_id="P1-T1",
                preferred_model="gpt-5.1-codex-max",
                run_id="2026-01-07_000000",
            )

        error_text = str(exc_info.value)
        assert permission_denied in error_text
        assert "copilot" in error_text.lower()
        assert "-p" in error_text or "--prompt" in error_text
        assert "--allow-tool" in error_text
        assert "write" in error_text
        assert "shell(poetry)" in error_text
        assert "shell(git)" in error_text

        # Sanity check the argv we actually attempted to run.
        assert captured_argv
        assert captured_argv[0] == str(fake_copilot)

    def test_run_copilot_reuses_session_when_requested(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() adds --session-path when resume_session=True and supported."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_copilot = fake_bin / "copilot"
        fake_copilot.write_text("#!/bin/sh\necho copilot")
        monkeypatch.setenv("PATH", str(fake_bin))
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: True,
        )

        captured_argv: list[str] = []

        class MockStdout:
            def read(self, size: int = -1) -> bytes:
                return b""

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                captured_argv.extend(argv)
                self.stdout = MockStdout()
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", MockPopen)

        log_file = tmp_path / "log" / "test.log"

        run_copilot(
            workspace=tmp_path,
            prompt_text="retry prompt",
            log_file=log_file,
            task_id="P1-T1",
            preferred_model=None,
            run_id="2026-01-07_000000",
            resume_session=True,
        )

        # Should reuse the same session path for resume
        assert "--session-path" in captured_argv
        session_idx = captured_argv.index("--session-path")
        assert captured_argv[session_idx + 1].endswith(
            "copilot_session_2026-01-07_000000_P1-T1.md"
        )

    def test_run_copilot_skips_session_when_not_supported(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() omits --session-path if CLI lacks support."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_copilot = fake_bin / "copilot"
        fake_copilot.write_text("#!/bin/sh\necho copilot")
        monkeypatch.setenv("PATH", str(fake_bin))
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        captured_argv: list[str] = []

        class MockStdout:
            def read(self, size: int = -1) -> bytes:
                return b""

        class MockPopen:
            def __init__(
                self, argv: list[str], *args: object, **kwargs: object
            ) -> None:
                captured_argv.extend(argv)
                self.stdout = MockStdout()
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", MockPopen)

        log_file = tmp_path / "log" / "test.log"

        run_copilot(
            workspace=tmp_path,
            prompt_text="retry prompt",
            log_file=log_file,
            task_id="P1-T1",
            preferred_model=None,
            run_id="2026-01-07_000000",
            resume_session=True,
        )

        assert "--session-path" not in captured_argv

    def test_run_copilot_times_out_when_cli_is_idle(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """run_copilot() terminates when Copilot CLI produces no output."""
        from scripts.dev_tools.atomic_executor.cli import run_copilot

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        copilot_exe = bin_dir / "copilot"
        copilot_exe.write_text("#!/bin/sh\nexit 0")
        copilot_exe.chmod(0o755)

        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{str(bin_dir)}{os.pathsep}{path}")
        monkeypatch.setenv("ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS", "0.2")
        monkeypatch.setattr(
            "scripts.dev_tools.atomic_executor.cli._copilot_supports_session",
            lambda exe: False,
        )

        stdout_r, stdout_w = os.pipe()
        os.close(stdout_w)
        stdout_stream = os.fdopen(stdout_r, "rb", buffering=0)

        hung_process_holder: dict[str, object] = {}

        class HungProcess:
            def __init__(self, argv: list[str]) -> None:
                self.args = argv
                self.stdout = stdout_stream
                self.returncode: int | None = None
                self.killed = False

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float | None = None) -> int | None:
                return self.returncode

            def kill(self) -> None:
                self.killed = True
                self.returncode = -9

        def fake_popen(argv: list[str], *args: object, **kwargs: object) -> HungProcess:
            proc = HungProcess(argv)
            hung_process_holder["proc"] = proc
            return proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        log_file = tmp_path / "log" / "test.log"

        with pytest.raises(TimeoutError):
            run_copilot(
                workspace=tmp_path,
                prompt_text="idle prompt",
                log_file=log_file,
                task_id="P1-T1",
                preferred_model=None,
                run_id="2026-01-07_000000",
            )

        proc = hung_process_holder.get("proc")
        assert proc is not None
        assert getattr(proc, "killed", False) is True
        stdout_stream.close()

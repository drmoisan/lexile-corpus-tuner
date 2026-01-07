"""Tests for resolve_execute_plan_prompt helper."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest

from scripts.dev_tools import resolve_execute_plan_prompt as module

FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "fixtures"
    / "resolve_execute_plan_prompt"
)


def test_replace_feature_token() -> None:
    prompt = "Hello <feature>"
    assert module.replace_feature_token(prompt, "abc") == "Hello abc"


def test_select_feature_folder_requested() -> None:
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    result = module.select_feature_folder(
        active_dir, "2025-12-18-docs-v3-upgrade", None
    )
    assert result == "2025-12-18-docs-v3-upgrade"


def test_select_feature_folder_branch_unique() -> None:
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    result = module.select_feature_folder(
        active_dir, None, "feature/docs-v3-upgrade-alt"
    )
    assert result == "2025-12-18-docs-v3-upgrade-alt"


def test_select_feature_folder_branch_ambiguous() -> None:
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    with pytest.raises(ValueError):
        module.select_feature_folder(active_dir, None, "docs-v3-upgrade")


def test_copy_to_clipboard_with_pyperclip(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPyperclip:
        def __init__(self) -> None:
            self.copied: str | None = None

        def copy(self, text: str) -> None:  # noqa: D401
            self.copied = text

    dummy = DummyPyperclip()
    monkeypatch.setitem(sys.modules, "pyperclip", dummy)

    assert module.copy_to_clipboard("hello") is True


def test_copy_to_clipboard_without_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    def _which(_name: str) -> str | None:
        return None

    monkeypatch.setattr(
        module.shutil, "which", cast(Callable[[str], str | None], _which)
    )

    assert module.copy_to_clipboard("hello") is False


def test_main_with_feature_prints_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    workspace = FIXTURE_ROOT
    code = module.main(
        [
            "--workspace",
            str(workspace),
            "--feature",
            "2025-12-18-docs-v3-upgrade",
            "--no-copy",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "2025-12-18-docs-v3-upgrade" in captured.out


def test_main_resolves_feature_from_branch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = FIXTURE_ROOT

    def _current_branch(_workspace: Path) -> str:
        return "feature/docs-v3-upgrade-alt"

    monkeypatch.setattr(
        module, "current_branch", cast(Callable[[Path], str], _current_branch)
    )

    code = module.main(
        [
            "--workspace",
            str(workspace),
            "--no-copy",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "2025-12-18-docs-v3-upgrade-alt" in captured.out


# Tests for helper functions


def test_read_text() -> None:
    """Test read_text reads file content."""
    mock_path = Mock(spec=Path)
    mock_path.read_text.return_value = "file contents"

    result = module.read_text(mock_path)

    assert result == "file contents"
    mock_path.read_text.assert_called_once_with(encoding="utf-8")


def test_current_branch_success() -> None:
    """Test current_branch returns branch name on success."""
    with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
        with patch.object(
            module.subprocess,
            "run",
            return_value=Mock(stdout="main\n"),
        ):
            result = module.current_branch(Path("/workspace"))
            assert result == "main"


def test_current_branch_no_git() -> None:
    """Test current_branch returns None when git is not available."""
    with patch.object(module.shutil, "which", return_value=None):
        result = module.current_branch(Path("/workspace"))
        assert result is None


def test_current_branch_subprocess_error() -> None:
    """Test current_branch returns None on subprocess error."""
    with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
        with patch.object(
            module.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            result = module.current_branch(Path("/workspace"))
            assert result is None


def test_current_branch_empty_output() -> None:
    """Test current_branch returns None when output is empty."""
    with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
        with patch.object(
            module.subprocess,
            "run",
            return_value=Mock(stdout=""),
        ):
            result = module.current_branch(Path("/workspace"))
            assert result is None


def test_normalize_branch_suffix_simple() -> None:
    """Test normalize_branch_suffix with simple branch."""
    assert module.normalize_branch_suffix("feature/my-feature") == "my-feature"


def test_normalize_branch_suffix_with_hash() -> None:
    """Test normalize_branch_suffix removes # symbols."""
    assert module.normalize_branch_suffix("feature/#42-fix") == "42-fix"


def test_normalize_branch_suffix_with_trailing_number() -> None:
    """Test normalize_branch_suffix removes trailing numbers."""
    assert module.normalize_branch_suffix("feature/fix-123") == "fix"


def test_normalize_branch_suffix_no_slash() -> None:
    """Test normalize_branch_suffix works without slash."""
    assert module.normalize_branch_suffix("main") == "main"


def test_list_feature_folders() -> None:
    """Test list_feature_folders returns sorted folder names."""
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    result = module.list_feature_folders(active_dir)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(name, str) for name in result)


def test_select_feature_folder_no_folders() -> None:
    """Test select_feature_folder raises when no folders exist."""
    with patch.object(module, "list_feature_folders", return_value=[]):
        with pytest.raises(ValueError, match="No feature folders found"):
            module.select_feature_folder(Path("/any"), None, None)


def test_select_feature_folder_requested_not_found() -> None:
    """Test select_feature_folder raises when requested folder not found."""
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    with pytest.raises(ValueError, match="not found"):
        module.select_feature_folder(active_dir, "nonexistent-feature", None)


def test_select_feature_folder_no_match() -> None:
    """Test select_feature_folder raises when no branch matches."""
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    with pytest.raises(ValueError, match="Could not resolve"):
        module.select_feature_folder(active_dir, None, "nonexistent-branch")


def test_build_prompt_text() -> None:
    """Test build_prompt_text loads and substitutes feature."""
    workspace = FIXTURE_ROOT
    prompt_path = workspace / ".github" / "prompts" / "execute-plan-template.md"
    result = module.build_prompt_text(workspace, "my-feature", prompt_path)
    assert "my-feature" in result
    assert "<feature>" not in result


def test_build_prompt_text_with_agent() -> None:
    """Test build_prompt_text substitutes agent token."""
    workspace = FIXTURE_ROOT
    prompt_path = workspace / ".github" / "prompts" / "execute-plan-template.md"
    # The template has "You are the “<agent>” execution agent."
    result = module.build_prompt_text(
        workspace, "my-feature", prompt_path, agent="Super Agent"
    )
    assert "Super Agent" in result
    assert "<agent>" not in result


def test_copy_to_clipboard_pyperclip_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test copy_to_clipboard falls back when pyperclip raises error."""

    class BrokenPyperclip:
        def copy(self, text: str) -> None:
            raise RuntimeError("Clipboard unavailable")

    monkeypatch.setitem(sys.modules, "pyperclip", BrokenPyperclip())

    def _which(_name: str) -> str | None:
        return None

    monkeypatch.setattr(
        module.shutil, "which", cast(Callable[[str], str | None], _which)
    )

    result = module.copy_to_clipboard("hello")
    assert result is False


def test_copy_to_clipboard_command_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test copy_to_clipboard uses command when available."""
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    def _which(name: str) -> str | None:
        return "/usr/bin/pbcopy" if name == "pbcopy" else None

    monkeypatch.setattr(
        module.shutil, "which", cast(Callable[[str], str | None], _which)
    )

    with patch.object(
        module.subprocess,
        "run",
        return_value=Mock(returncode=0),
    ):
        result = module.copy_to_clipboard("hello")
        assert result is True


def test_copy_to_clipboard_command_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test copy_to_clipboard continues when command fails."""
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    def _which(name: str) -> str | None:
        return "/usr/bin/pbcopy" if name == "pbcopy" else None

    monkeypatch.setattr(
        module.shutil, "which", cast(Callable[[str], str | None], _which)
    )

    with patch.object(
        module.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, "pbcopy"),
    ):
        result = module.copy_to_clipboard("hello")
        assert result is False


def test_parse_args_with_feature() -> None:
    """Test parse_args with feature argument."""
    args = module.parse_args(["--feature", "my-feature"])
    assert args.feature == "my-feature"
    assert args.no_copy is False


def test_parse_args_with_no_copy() -> None:
    """Test parse_args with no-copy flag."""
    args = module.parse_args(["--no-copy"])
    assert args.no_copy is True


def test_parse_args_with_workspace() -> None:
    """Test parse_args with workspace argument."""
    args = module.parse_args(["--workspace", "/custom/path"])
    assert args.workspace == "/custom/path"


def test_parse_args_with_prompt_path() -> None:
    """Test parse_args with prompt-path argument."""
    args = module.parse_args(["--prompt-path", "custom.md"])
    assert args.prompt_path == "custom.md"


def test_parse_args_defaults() -> None:
    """Test parse_args with no arguments uses defaults."""
    args = module.parse_args([])
    assert args.feature is None
    assert args.no_copy is False
    assert args.workspace is None


def test_main_prompt_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """Test main returns error when prompt file not found."""
    code = module.main(
        [
            "--workspace",
            str(FIXTURE_ROOT),
            "--feature",
            "2025-12-18-docs-v3-upgrade",
            "--prompt-path",
            "nonexistent.md",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err


def test_main_feature_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """Test main returns error when feature not found."""
    code = module.main(
        [
            "--workspace",
            str(FIXTURE_ROOT),
            "--feature",
            "nonexistent-feature",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err


def test_main_with_clipboard_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main with successful clipboard copy."""

    class DummyPyperclip:
        def copy(self, text: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "pyperclip", DummyPyperclip())

    workspace = FIXTURE_ROOT
    code = module.main(
        [
            "--workspace",
            str(workspace),
            "--feature",
            "2025-12-18-docs-v3-upgrade",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "copied to clipboard" in captured.err.lower()


def test_main_clipboard_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main prints message when clipboard unavailable."""
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    def _which(_name: str) -> str | None:
        return None

    monkeypatch.setattr(
        module.shutil, "which", cast(Callable[[str], str | None], _which)
    )

    workspace = FIXTURE_ROOT
    code = module.main(
        [
            "--workspace",
            str(workspace),
            "--feature",
            "2025-12-18-docs-v3-upgrade",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "not available" in captured.err.lower()


def test_select_feature_folder_is_path() -> None:
    """Test select_feature_folder resolves a full file path to the feature folder."""
    active_dir = FIXTURE_ROOT / "docs" / "features" / "active"
    feature_name = "2025-12-18-docs-v3-upgrade"

    # Construct a path that points inside the feature folder
    feature_folder_path = active_dir / feature_name
    file_path = feature_folder_path / "plan.md"

    # Pass as string, mimicking what happens when user selects a file in VS Code input
    result = module.select_feature_folder(active_dir, str(file_path), None)
    assert result == feature_name

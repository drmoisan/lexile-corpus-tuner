"""Tests for resolve_execute_plan_prompt helper."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

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

"""Tests for new_potential_bug_entry Python rewrite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from scripts.dev_tools import new_potential_bug_entry as mod

if TYPE_CHECKING:
    from collections.abc import Iterable


class FakeFileSystem(mod.FileSystem):
    def __init__(self) -> None:
        self.files: dict[Path, str] = {}
        self.dirs: set[Path] = set()

    def ensure_dir(self, path: Path) -> None:
        self.dirs.add(path)

    def copy_file(self, src: Path, dest: Path) -> None:
        if src not in self.files:
            raise FileNotFoundError(src)
        self.files[dest] = self.files[src]

    def read_text(self, path: Path) -> str:
        return self.files[path]

    def write_text(self, path: Path, content: str) -> None:
        self.files[path] = content


def test_default_git_config_lookup_returns_none_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(_name: str) -> None:
        return None

    monkeypatch.setattr(mod.shutil, "which", missing_git)
    assert mod.default_git_config_lookup("user.name") is None


def test_default_env_lookup_returns_none_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USERNAME", "")
    assert mod.default_env_lookup("USERNAME") is None


def test_get_author_falls_back_to_unknown() -> None:
    assert mod.get_author(lambda _: None, lambda _: None) == "Unknown"


def test_validate_short_name_accepts_valid() -> None:
    mod.validate_short_name("api-timeout")


def test_validate_short_name_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        mod.validate_short_name("Invalid Name")


def test_render_content_replaces_placeholders() -> None:
    template = "<bug-name> on YYYY-MM-DD by - Author: name"
    rendered = mod.render_content(template, "api-timeout", "2025-12-15", "Jane")
    assert rendered == "api-timeout on 2025-12-15 by - Author: Jane"


def test_default_code_launcher_runs_when_code_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:  # noqa: ARG001
        launched.append(cmd)

    def code_available(_name: str) -> str:
        return "/usr/bin/code"

    monkeypatch.setattr(mod.shutil, "which", code_available)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod.default_code_launcher([Path("file.md")]) is True
    assert launched[0][0] == "/usr/bin/code"


def test_default_code_launcher_returns_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def code_missing(_name: str) -> None:
        return None

    monkeypatch.setattr(mod.shutil, "which", code_missing)
    assert mod.default_code_launcher([Path("file.md")]) is False


def test_create_bug_entry_writes_file_and_launches_code() -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    template_path = workspace / "docs/features/templates/bug/potential_bug.md"
    target_dir = workspace / "docs/features/potential"
    target_path = target_dir / "2025-12-15-api-timeout.md"

    fs.files[template_path] = "<bug-name> on YYYY-MM-DD by - Author: name"
    launched: list[list[Path]] = []

    def launcher(files: Iterable[Path]) -> bool:
        launched.append(list(files))
        return True

    created = mod.create_bug_entry(
        short_name="api-timeout",
        workspace=workspace,
        fs=fs,
        author_provider=lambda: "Jane Doe",
        code_launcher=launcher,
        entry_date="2025-12-15",
    )

    assert created == target_path
    assert fs.files[target_path] == "api-timeout on 2025-12-15 by - Author: Jane Doe"
    assert launched == [[target_path]]


def test_create_bug_entry_warns_when_code_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fs = FakeFileSystem()
    workspace = Path("/workspace")
    template_path = workspace / "docs/features/templates/bug/potential_bug.md"
    target_path = workspace / "docs/features/potential/2025-12-15-api-timeout.md"
    fs.files[template_path] = "<bug-name> on YYYY-MM-DD"

    created = mod.create_bug_entry(
        short_name="api-timeout",
        workspace=workspace,
        fs=fs,
        author_provider=lambda: "Jane",
        code_launcher=lambda files: False,
        entry_date="2025-12-15",
    )

    captured = capsys.readouterr()
    assert "WARNING: VS Code" in captured.out
    assert str(target_path) in captured.out
    assert created == target_path


def test_real_filesystem_methods_are_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_mkdir(self: Path, parents: bool, exist_ok: bool) -> None:  # noqa: ARG002
        calls.append("mkdir")

    def fake_copyfile(src: Path, dest: Path) -> None:
        calls.append(f"copy:{src}->{dest}")

    def fake_read_text(self: Path, encoding: str) -> str:  # noqa: ARG002
        calls.append("read_text")
        return "template"

    def fake_write_text(
        self: Path, content: str, encoding: str
    ) -> None:  # noqa: ARG002
        calls.append("write_text")

    monkeypatch.setattr(Path, "mkdir", fake_mkdir, raising=True)
    monkeypatch.setattr(mod.shutil, "copyfile", fake_copyfile)
    monkeypatch.setattr(Path, "read_text", fake_read_text, raising=True)
    monkeypatch.setattr(Path, "write_text", fake_write_text, raising=True)

    fs = mod.RealFileSystem()
    workspace = Path("/workspace")
    target_dir = workspace / "docs/features/potential"
    target = target_dir / "today-test.md"
    template = workspace / "docs/features/templates/bug/potential_bug.md"

    fs.ensure_dir(target_dir)
    fs.copy_file(template, target)
    _ = fs.read_text(target)
    fs.write_text(target, "x")

    assert calls[:1] == ["mkdir"]
    assert any(call.startswith("copy:") for call in calls)
    assert "read_text" in calls
    assert "write_text" in calls


def test_main_exits_on_invalid_short_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "parse_args", lambda: argparse.Namespace(short_name="Invalid Name")
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 1


def test_main_exits_on_missing_template(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_bug_entry(*args: object, **kwargs: Any) -> NoReturn:  # noqa: ARG001
        raise FileNotFoundError("missing template")

    monkeypatch.setattr(
        mod, "parse_args", lambda: argparse.Namespace(short_name="api-timeout")
    )
    monkeypatch.setattr(mod, "create_bug_entry", fake_create_bug_entry)

    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 1

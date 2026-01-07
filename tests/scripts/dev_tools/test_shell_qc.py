from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import pytest  # noqa: TCH002  # Needed at runtime for pytest fixtures

import scripts.dev_tools.shell_qc as shell_qc


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "shell_qc"


def test_discover_shell_scripts_includes_expected() -> None:
    root = _fixture_root()

    discovered = shell_qc.discover_shell_scripts(root)

    expected = {
        root / "tools" / "format_me.sh",
        root / "scripts" / "with_shebang",
    }
    assert set(discovered) == expected
    assert root / "scripts" / "ignored.txt" not in discovered
    assert root / "scripts" / "pwsh_script.ps1" not in discovered
    assert root / "scripts" / "node_modules" / "skip.sh" not in discovered


def test_find_bats_test_dirs_detects_shell() -> None:
    root = _fixture_root()

    assert shell_qc.find_bats_test_dirs(root) == [root / "tests" / "shell"]


def test_run_check_skips_when_no_scripts(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return []

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)

    exit_code = shell_qc.run_check()

    assert exit_code == 0
    assert "No shell scripts found; skipping." in capsys.readouterr().out


def test_run_check_missing_tool(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return [Path("tools/test.sh")]

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)

    def _which(tool: str) -> str | None:
        return None if tool == "shfmt" else "/usr/bin/shellcheck"

    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_check()

    output = capsys.readouterr().out
    assert exit_code == 127
    assert "Missing required tool: shfmt" in output
    assert "apt-get install -y shfmt" in output
    assert "brew install shfmt" in output
    assert "WSL" in output


def test_run_check_runs_shellcheck_per_file(monkeypatch: MonkeyPatch) -> None:
    files = [Path("tools/a.sh"), Path("scripts/b.sh")]

    def _discover(_: Path | None = None) -> list[Path]:
        return files

    def _which(_: str) -> str:
        return "/usr/bin/tool"

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        if command[0] == "shellcheck" and "tools" in command[1]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_check()

    assert exit_code == 1
    assert calls[0] == [
        "shfmt",
        "-d",
        str(Path("tools/a.sh")),
        str(Path("scripts/b.sh")),
    ]
    shellcheck_calls = [call for call in calls if call[0] == "shellcheck"]
    assert shellcheck_calls == [
        ["shellcheck", str(Path("tools/a.sh"))],
        ["shellcheck", str(Path("scripts/b.sh"))],
    ]


def test_run_format_missing_tool(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return [Path("tools/a.sh")]

    def _which(_: str) -> str | None:
        return None

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_format()

    assert exit_code == 127
    assert "Missing required tool: shfmt" in capsys.readouterr().out


def test_run_format_invokes_shfmt(monkeypatch: MonkeyPatch) -> None:
    files = [Path("tools/a.sh")]

    def _discover(_: Path | None = None) -> list[Path]:
        return files

    def _which(_: str) -> str:
        return "/usr/bin/shfmt"

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_format()

    assert exit_code == 0
    assert calls == [["shfmt", "-w", str(Path("tools/a.sh"))]]


def test_run_test_skips_when_no_dirs(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return []

    def _which(_: str) -> str:
        return "/usr/bin/bats"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert "No shell test directories found; skipping." in capsys.readouterr().out


def test_run_test_skips_when_bats_missing(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(_: str) -> str | None:
        return None

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert "bats not installed" in capsys.readouterr().out


def test_run_test_runs_bats(monkeypatch: MonkeyPatch) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(_: str) -> str:
        return "/usr/bin/bats"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert calls == [["bats", str(Path("tests/shell"))]]


def test_parse_args_accepts_commands() -> None:
    assert shell_qc.parse_args(["check"]).command == "check"
    assert shell_qc.parse_args(["format"]).command == "format"
    assert shell_qc.parse_args(["test"]).command == "test"


def test_main_dispatches_to_check(monkeypatch: MonkeyPatch) -> None:
    def _run_check(_: Path | None = None) -> int:
        return 0

    monkeypatch.setattr(shell_qc, "run_check", _run_check)

    assert shell_qc.main(["check"]) == 0


def test_main_dispatches_to_format(monkeypatch: MonkeyPatch) -> None:
    def _run_format(_: Path | None = None) -> int:
        return 0

    monkeypatch.setattr(shell_qc, "run_format", _run_format)

    assert shell_qc.main(["format"]) == 0


def test_main_dispatches_to_test(monkeypatch: MonkeyPatch) -> None:
    def _run_test(_: Path | None = None) -> int:
        return 0

    monkeypatch.setattr(shell_qc, "run_test", _run_test)

    assert shell_qc.main(["test"]) == 0


def test_main_check_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_check(root: Path | None = None) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_check", _run_check)

    assert shell_qc.main_check() == 0
    assert called["hit"] is True


def test_main_format_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_format(root: Path | None = None) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_format", _run_format)

    assert shell_qc.main_format() == 0
    assert called["hit"] is True


def test_main_test_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_test(root: Path | None = None) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_test", _run_test)

    assert shell_qc.main_test() == 0
    assert called["hit"] is True

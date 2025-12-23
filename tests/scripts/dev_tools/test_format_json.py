from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import scripts.dev_tools.format_json as fmt


def _patch_io(monkeypatch: MonkeyPatch, store: dict[Path, str]) -> None:
    def read_text(self: Path, *args: Any, **kwargs: Any):
        return store[self]

    def write_text(self: Path, data: str, *args: Any, **kwargs: Any):
        store[self] = data
        return len(data)

    def _is_file(self: Path) -> bool:
        return True

    monkeypatch.setattr(Path, "read_text", read_text, raising=False)
    monkeypatch.setattr(Path, "write_text", write_text, raising=False)
    monkeypatch.setattr(Path, "is_file", _is_file, raising=False)


def _fake_run(stdout: str = "{}\n", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_format_no_change(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): "{}\n"}
    _patch_io(monkeypatch, store)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("{}\n", 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)

    result = fmt.format_files([Path("/f.json")], check=False, verbose=True)

    assert result.changed is False
    assert result.failed is False
    assert "already formatted" in result.messages[0]


def test_format_rewrites(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): '{"b":1}\n'}
    _patch_io(monkeypatch, store)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run('{\n  "b": 1\n}\n', 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)

    result = fmt.format_files([Path("/f.json")], check=False, verbose=True)

    assert result.changed is True
    assert result.failed is False
    assert store[Path("/f.json")] == '{\n  "b": 1\n}\n'


def test_format_check_mode(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): '{"b":1}\n'}
    _patch_io(monkeypatch, store)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run('{\n  "b": 1\n}\n', 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)

    result = fmt.format_files([Path("/f.json")], check=True, verbose=True)

    assert result.changed is True
    assert result.failed is False
    assert store[Path("/f.json")] == '{"b":1}\n'  # unchanged in check mode


def test_format_jq_failure(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): '{"b":1}\n'}
    _patch_io(monkeypatch, store)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("", returncode=1)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)

    result = fmt.format_files([Path("/f.json")], check=False, verbose=True)

    assert result.failed is True
    assert result.changed is False
    assert any("jq failed" in m for m in result.messages)

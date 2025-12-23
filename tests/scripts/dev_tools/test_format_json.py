from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import pytest  # noqa: TCH002  # Needed at runtime for pytest decorators

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


def test_format_result_init() -> None:
    """FormatResult should initialize with provided values."""
    result = fmt.FormatResult(True, False, ["message1", "message2"])
    assert result.changed is True
    assert result.failed is False
    assert result.messages == ["message1", "message2"]


def test_format_files_jq_not_found(monkeypatch: MonkeyPatch) -> None:
    """format_files should handle missing jq executable."""

    def _which(_: str) -> None:
        return None

    monkeypatch.setattr(fmt.shutil, "which", _which)

    result = fmt.format_files([Path("/f.json")], check=False, verbose=False)

    assert result.failed is True
    assert result.changed is False
    assert "jq executable not found on PATH" in result.messages[0]


def test_format_files_skips_non_files(monkeypatch: MonkeyPatch) -> None:
    """format_files should skip paths that are not files."""

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _is_file(self: Path) -> bool:
        return False

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(Path, "is_file", _is_file)

    result = fmt.format_files([Path("/dir")], check=False, verbose=True)

    assert result.changed is False
    assert result.failed is False
    assert result.messages == []


def test_format_files_non_verbose_hides_unchanged(monkeypatch: MonkeyPatch) -> None:
    """format_files should not report unchanged files when not verbose."""
    store: dict[Path, str] = {Path("/f.json"): "{}\n"}
    _patch_io(monkeypatch, store)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("{}\n", 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)

    result = fmt.format_files([Path("/f.json")], check=False, verbose=False)

    assert result.changed is False
    assert result.failed is False
    assert result.messages == []


def test_run_jq_format_with_error_message(monkeypatch: MonkeyPatch) -> None:
    """run_jq_format should capture stderr on failure."""
    store: dict[Path, str] = {Path("/f.json"): "invalid"}
    _patch_io(monkeypatch, store)

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(stdout="", stderr="parse error", returncode=1)

    monkeypatch.setattr(fmt.subprocess, "run", _run)

    changed, failed, msg = fmt.run_jq_format(Path("/f.json"), False, "/usr/bin/jq")

    assert changed is False
    assert failed is True
    assert "parse error" in msg


def test_parse_args_defaults() -> None:
    """parse_args with no arguments should use defaults."""
    args = fmt.parse_args([])
    assert args.paths == []
    assert args.check is False
    assert args.verbose is False


def test_parse_args_with_paths() -> None:
    """parse_args should accept path arguments."""
    args = fmt.parse_args(["file1.json", "file2.json"])
    assert args.paths == ["file1.json", "file2.json"]


def test_parse_args_check_flag() -> None:
    """parse_args should accept --check flag."""
    args = fmt.parse_args(["--check"])
    assert args.check is True


def test_parse_args_verbose_flag() -> None:
    """parse_args should accept --verbose flag."""
    args = fmt.parse_args(["--verbose"])
    assert args.verbose is True


def test_parse_args_combined() -> None:
    """parse_args should handle multiple flags and paths."""
    args = fmt.parse_args(["--check", "--verbose", "test.json"])
    assert args.check is True
    assert args.verbose is True
    assert args.paths == ["test.json"]


def test_main_no_paths_uses_governed(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main with no paths should use iter_governed_files."""
    json_file = tmp_path / "test.json"
    json_file.write_text("{}")

    def mock_iter(_: Path) -> list[Path]:
        return [json_file]

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("{}\n", 0)

    monkeypatch.setattr(fmt, "iter_governed_files", mock_iter)
    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main([])
    assert exit_code == 0


def test_main_with_file_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main should format specific file when path provided."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"b":1}')

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run('{\n  "b": 1\n}\n', 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main([str(json_file)])
    assert exit_code == 0


def test_main_with_directory_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main should recursively find JSON files in directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    json_file = subdir / "test.json"
    json_file.write_text("{}")

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("{}\n", 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main([str(tmp_path)])
    assert exit_code == 0


def test_main_check_mode_exits_1_on_changes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """main in check mode should return 1 when changes needed."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"b":1}')

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run('{\n  "b": 1\n}\n', 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main(["--check", str(json_file)])
    assert exit_code == 1


def test_main_failure_exits_1(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main should return 1 on formatting failures."""
    json_file = tmp_path / "test.json"
    json_file.write_text("invalid")

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run("", returncode=1)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main([str(json_file)])
    assert exit_code == 1


def test_main_verbose_mode_already_formatted(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main with --verbose should print 'already formatted' when no changes."""
    json_file = tmp_path / "test.json"
    original = '{\n  "b": 1\n}\n'
    json_file.write_text(original)

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run(original, 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main(["--verbose", str(json_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "already formatted" in captured.out


def test_main_verbose_mode_reformatted(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main with --verbose should print 'reformatted' when file is changed."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"b":1}')

    def _which(_: str) -> str:
        return "/usr/bin/jq"

    def _run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_run('{\n  "b": 1\n}\n', 0)

    monkeypatch.setattr(fmt.shutil, "which", _which)
    monkeypatch.setattr(fmt.subprocess, "run", _run)
    monkeypatch.setattr(sys, "argv", ["format_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "format_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "format_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = fmt.main(["--verbose", str(json_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "reformatted" in captured.out

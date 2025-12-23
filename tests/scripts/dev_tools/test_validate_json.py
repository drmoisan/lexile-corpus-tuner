from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import pytest

import scripts.dev_tools.validate_json as val


def _patch_read(monkeypatch: MonkeyPatch, store: dict[Path, str]) -> None:
    def read_text(self: Path, *args: Any, **kwargs: Any):
        return store[self]

    monkeypatch.setattr(Path, "read_text", read_text, raising=False)


def test_validate_result_init() -> None:
    """ValidateResult should initialize with empty state."""
    result = val.ValidateResult()
    assert result.failed is False
    assert result.messages == []


def test_cache_path_generates_deterministic_hash() -> None:
    """_cache_path should generate deterministic SHA256-based filenames."""
    cache_dir = Path("/cache")
    uri = "https://example.com/schema.json"
    path1 = val._cache_path(cache_dir, uri)  # type: ignore[reportPrivateUsage]
    path2 = val._cache_path(cache_dir, uri)  # type: ignore[reportPrivateUsage]
    assert path1 == path2
    assert path1.parent == cache_dir
    assert path1.suffix == ".json"


def test_load_schema_from_cache(tmp_path: Path) -> None:
    """_load_schema should load from cache if present."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    uri = "https://example.com/schema.json"
    cache_file = val._cache_path(cache_dir, uri)  # type: ignore[reportPrivateUsage]
    cache_file.write_text('{"type": "object"}')

    schema = val._load_schema(uri, cache_dir)  # type: ignore[reportPrivateUsage]
    assert schema == {"type": "object"}


def test_load_schema_unsupported_scheme(tmp_path: Path) -> None:
    """_load_schema should reject unsupported URI schemes."""
    with pytest.raises(ValueError, match="Unsupported schema URI scheme"):
        val._load_schema("ftp://example.com/schema.json", tmp_path / "cache")  # type: ignore[reportPrivateUsage]


def test_load_schema_missing_scheme(tmp_path: Path) -> None:
    """_load_schema should reject URIs with missing scheme."""
    with pytest.raises(ValueError, match="Unsupported schema URI scheme"):
        val._load_schema("no-scheme-here", tmp_path / "cache")  # type: ignore[reportPrivateUsage]


def test_load_schema_relative_path(tmp_path: Path) -> None:
    """_load_schema should resolve relative file paths against the source file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        '{"type": "object", "properties": {"key": {"type": "number"}}}'
    )
    source_path = tmp_path / "data.json"

    schema = val._load_schema("./schema.json", cache_dir, source_path)  # type: ignore[reportPrivateUsage]

    assert schema == {"type": "object", "properties": {"key": {"type": "number"}}}


def test_load_schema_fetch_and_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """_load_schema should fetch remote schema and cache it."""
    cache_dir = tmp_path / "cache"
    uri = "https://example.com/schema.json"
    schema_content = '{"type": "object", "properties": {}}'

    mock_response = MagicMock()
    mock_response.read.return_value = schema_content.encode("utf-8")
    mock_response.__enter__ = lambda self: self  # type: ignore[reportUnknownLambdaType]
    mock_response.__exit__ = lambda self, *args: None  # type: ignore[reportUnknownLambdaType]

    def mock_urlopen(url: str) -> MagicMock:
        return mock_response

    monkeypatch.setattr(val.urllib.request, "urlopen", mock_urlopen)

    schema = val._load_schema(uri, cache_dir)  # type: ignore[reportPrivateUsage]
    assert schema == {"type": "object", "properties": {}}

    cache_file = val._cache_path(cache_dir, uri)  # type: ignore[reportPrivateUsage]
    assert cache_file.exists()
    assert cache_file.read_text() == schema_content


def test_validate_ok(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {
        Path("/f.json"): '{"$schema":"https://example.com/schema.json","key":1}'
    }
    _patch_read(monkeypatch, store)

    def _schema(_: str, __: Path, ___: Path | None = None) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"key": {"type": "number"}},
            "required": ["key"],
        }

    monkeypatch.setattr(val, "_load_schema", _schema)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is True
    assert "ok" in msg


def test_validate_relative_schema(tmp_path: Path) -> None:
    """validate_file should resolve relative $schema paths."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"key": {"type": "number"}},
                "required": ["key"],
            }
        )
    )

    data_path = tmp_path / "data.json"
    data_path.write_text('{"$schema": "./schema.json", "key": 1}')

    ok, msg = val.validate_file(data_path, tmp_path / "cache")

    assert ok is True
    assert "ok" in msg


def test_validate_missing_schema(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): '{"key":1}'}
    _patch_read(monkeypatch, store)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "missing $schema" in msg


def test_validate_schema_failure(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {
        Path("/f.json"): '{"$schema":"https://example.com/schema.json","key":"bad"}'
    }
    _patch_read(monkeypatch, store)

    def _schema(_: str, __: Path, ___: Path | None = None) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"key": {"type": "number"}},
            "required": ["key"],
        }

    monkeypatch.setattr(val, "_load_schema", _schema)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "schema validation failed" in msg


def test_validate_bad_json(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {Path("/f.json"): '{"key":'}
    _patch_read(monkeypatch, store)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "invalid JSON" in msg


def test_validate_non_dict_root(monkeypatch: MonkeyPatch) -> None:
    """JSON arrays or primitives at root should fail validation."""
    store: dict[Path, str] = {Path("/f.json"): '["array"]'}
    _patch_read(monkeypatch, store)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "JSON root must be an object" in msg


def test_validate_schema_not_string(monkeypatch: MonkeyPatch) -> None:
    """$schema that is not a string should fail."""
    store: dict[Path, str] = {Path("/f.json"): '{"$schema": 123}'}
    _patch_read(monkeypatch, store)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "missing $schema" in msg


def test_validate_exception_during_validation(monkeypatch: MonkeyPatch) -> None:
    """Exceptions during validation should be caught and reported."""
    store: dict[Path, str] = {
        Path("/f.json"): '{"$schema":"https://example.com/schema.json"}'
    }
    _patch_read(monkeypatch, store)

    def _schema(_: str, __: Path, ___: Path | None = None) -> dict[str, object]:
        raise RuntimeError("Schema load failed")

    monkeypatch.setattr(val, "_load_schema", _schema)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

    assert ok is False
    assert "validation error" in msg
    assert "Schema load failed" in msg


def test_parse_args_defaults() -> None:
    """parse_args with no arguments should use defaults."""
    args = val.parse_args([])
    assert args.paths == []
    assert args.verbose is False
    assert args.cache_dir == ".cache/schemas"


def test_parse_args_with_paths() -> None:
    """parse_args should accept path arguments."""
    args = val.parse_args(["file1.json", "file2.json"])
    assert args.paths == ["file1.json", "file2.json"]


def test_parse_args_verbose() -> None:
    """parse_args should accept --verbose flag."""
    args = val.parse_args(["--verbose"])
    assert args.verbose is True


def test_parse_args_custom_cache_dir() -> None:
    """parse_args should accept --cache-dir argument."""
    args = val.parse_args(["--cache-dir", "/custom/cache"])
    assert args.cache_dir == "/custom/cache"


def test_collect_targets_with_file_paths(tmp_path: Path) -> None:
    """collect_targets should collect specific file paths."""
    file1 = tmp_path / "test1.json"
    file1.write_text("{}")
    file2 = tmp_path / "test2.json"
    file2.write_text("{}")

    targets = val.collect_targets(tmp_path, [str(file1), str(file2)])
    assert file1 in targets
    assert file2 in targets


def test_collect_targets_with_directory(tmp_path: Path) -> None:
    """collect_targets should recursively find JSON files in directories."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    json_file = subdir / "test.json"
    json_file.write_text("{}")

    targets = val.collect_targets(tmp_path, [str(tmp_path)])
    assert json_file in targets


def test_collect_targets_defaults_to_governed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """collect_targets with no paths should use iter_governed_files."""
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    tasks_json = vscode_dir / "tasks.json"
    tasks_json.write_text("{}")

    def mock_iter(_: Path) -> list[Path]:
        return [tasks_json]

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)

    targets = val.collect_targets(tmp_path, [])
    assert tasks_json in targets


def test_main_no_files(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main with no matching files should return 0."""

    def mock_iter(_: Path) -> list[Path]:
        return []

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)
    monkeypatch.setattr(sys, "argv", ["validate_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "validate_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "validate_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = val.main([])
    assert exit_code == 0


def test_main_all_valid(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main with all valid files should return 0."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"$schema":"https://example.com/schema.json"}')

    def mock_iter(_: Path) -> list[Path]:
        return [json_file]

    def _schema(_: str, __: Path, ___: Path | None = None) -> dict[str, object]:
        return {"type": "object"}

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)
    monkeypatch.setattr(val, "_load_schema", _schema)
    monkeypatch.setattr(sys, "argv", ["validate_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "validate_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "validate_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = val.main([])
    assert exit_code == 0


def test_main_validation_failure(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main with validation failures should return 1."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"key":1}')

    def mock_iter(_: Path) -> list[Path]:
        return [json_file]

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)
    monkeypatch.setattr(sys, "argv", ["validate_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "validate_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "validate_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = val.main([])
    assert exit_code == 1


def test_main_verbose_mode(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main with --verbose should print status for all files."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"$schema":"https://example.com/schema.json"}')

    def mock_iter(_: Path) -> list[Path]:
        return [json_file]

    def _schema(_: str, __: Path, ___: Path | None = None) -> dict[str, object]:
        return {"type": "object"}

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)
    monkeypatch.setattr(val, "_load_schema", _schema)
    monkeypatch.setattr(sys, "argv", ["validate_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "validate_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "validate_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = val.main(["--verbose"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "ok" in captured.out


def test_main_custom_cache_dir(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """main should respect --cache-dir argument."""
    json_file = tmp_path / "test.json"
    json_file.write_text('{"$schema":"https://example.com/schema.json"}')

    custom_cache = tmp_path / "custom_cache"

    def mock_iter(_: Path) -> list[Path]:
        return [json_file]

    def _schema(
        uri: str, cache_dir: Path, base_path: Path | None = None
    ) -> dict[str, object]:
        assert cache_dir == custom_cache
        assert base_path == json_file
        return {"type": "object"}

    monkeypatch.setattr(val, "iter_governed_files", mock_iter)
    monkeypatch.setattr(val, "_load_schema", _schema)
    monkeypatch.setattr(sys, "argv", ["validate_json.py"])

    original_resolve = Path.resolve

    def mock_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if "validate_json.py" in str(self):
            return tmp_path / "scripts" / "dev_tools" / "validate_json.py"
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    exit_code = val.main(["--cache-dir", str(custom_cache)])
    assert exit_code == 0

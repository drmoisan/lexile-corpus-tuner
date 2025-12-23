from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import scripts.dev_tools.validate_json as val


def _patch_read(monkeypatch: MonkeyPatch, store: dict[Path, str]) -> None:
    def read_text(self: Path, *args: Any, **kwargs: Any):
        return store[self]

    monkeypatch.setattr(Path, "read_text", read_text, raising=False)


def test_validate_ok(monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, str] = {
        Path("/f.json"): '{"$schema":"https://example.com/schema.json","key":1}'
    }
    _patch_read(monkeypatch, store)

    def _schema(_: str, __: Path) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"key": {"type": "number"}},
            "required": ["key"],
        }

    monkeypatch.setattr(val, "_load_schema", _schema)

    ok, msg = val.validate_file(Path("/f.json"), Path("/cache"))

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

    def _schema(_: str, __: Path) -> dict[str, object]:
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

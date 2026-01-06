from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)

FileCache = enrich.FileCache
MatchResult = enrich.MatchResult


class _FakeHandle:
    def __init__(self, store: dict[str, str], key: str, mode: str) -> None:
        self._store = store
        self._key = key
        self._mode = mode
        self._buffer: list[str] = []

    def __enter__(self) -> _FakeHandle:
        return self

    def __exit__(self, *_args: Any) -> None:
        if "w" in self._mode:
            self._store[self._key] = "".join(self._buffer)

    def read(self) -> str:
        return self._store[self._key]

    def write(self, data: str) -> None:
        self._buffer.append(data)


class _FakePath:
    def __init__(self, key: str, store: dict[str, str]) -> None:
        self._key = key
        self._store = store

    def exists(self) -> bool:
        return self._key in self._store

    def open(self, mode: str, encoding: str) -> _FakeHandle:  # noqa: ARG002
        return _FakeHandle(self._store, self._key, mode)


class _FakeDir:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def mkdir(self, parents: bool, exist_ok: bool) -> None:  # noqa: ARG002
        return None

    def __truediv__(self, name: str) -> _FakePath:
        return _FakePath(name, self._store)


def test_file_cache_round_trip_uses_json_payload() -> None:
    store: dict[str, str] = {}
    cache = FileCache(cast(Path, _FakeDir(store)))
    cache.set("sample", MatchResult(year=1999, confidence="high", source="openlibrary"))

    result = cache.get("sample")

    assert result is not None
    assert result.year == 1999
    assert result.confidence == "high"
    assert result.source == "openlibrary"
    assert any("confidence" in content for content in store.values())


def test_file_cache_returns_none_when_missing() -> None:
    cache = FileCache(cast(Path, _FakeDir({})))

    result = cache.get("missing")

    assert result is None


def test_file_cache_returns_none_on_invalid_json() -> None:
    store = {"bad": "not-json"}
    cache = FileCache(cast(Path, _FakeDir(store)))

    result = cache.get("bad")

    assert result is None

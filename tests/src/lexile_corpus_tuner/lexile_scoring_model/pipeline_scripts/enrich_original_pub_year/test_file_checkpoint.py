from __future__ import annotations

import json
from typing import Any, cast

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    enrich_original_pub_year as enrich,
)

FileCheckpoint = enrich.FileCheckpoint
Summary = enrich.Summary


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

    @property
    def parent(self) -> _FakePath:
        return self

    def mkdir(self, parents: bool, exist_ok: bool) -> None:  # noqa: ARG002
        return None

    def exists(self) -> bool:
        return self._key in self._store

    def open(self, mode: str, encoding: str) -> _FakeHandle:  # noqa: ARG002
        return _FakeHandle(self._store, self._key, mode)


def test_load_returns_zero_when_missing_or_invalid() -> None:
    store: dict[str, str] = {}
    checkpoint = FileCheckpoint(cast(enrich.Path, _FakePath("ckpt.json", store)))

    assert checkpoint.load() == 0

    store["ckpt.json"] = "not-json"
    assert checkpoint.load() == 0


def test_load_returns_last_index_from_file() -> None:
    payload = json.dumps({"last_index": 42})
    store = {"ckpt.json": payload}
    checkpoint = FileCheckpoint(cast(enrich.Path, _FakePath("ckpt.json", store)))

    assert checkpoint.load() == 42


def test_save_writes_summary_payload() -> None:
    store: dict[str, str] = {}
    checkpoint = FileCheckpoint(cast(enrich.Path, _FakePath("ckpt.json", store)))
    summary = Summary()
    summary.matched_high = 1
    summary.matched_low = 2
    summary.matched_none = 3
    summary.errors = 4

    checkpoint.save(10, summary)

    assert "ckpt.json" in store
    data = json.loads(store["ckpt.json"])
    assert data["last_index"] == 10
    assert data["summary"] == summary.to_dict()

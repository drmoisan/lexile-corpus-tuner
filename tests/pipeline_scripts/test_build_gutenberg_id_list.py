from __future__ import annotations

# ruff: noqa: TC006  # allow explicit typing casts for test doubles
# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import requests
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    build_gutenberg_id_list as bgid,
)

from .conftest import InMemoryParquetStore, InMemoryPath, make_response


def test_file_parquet_store_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    storage: dict[str, str] = {}
    parquet_path = InMemoryPath("data/file.parquet", storage)
    loaded_df = pd.DataFrame({"id": [1]})
    read_calls: list[Path] = []
    write_calls: list[tuple[Path, bool]] = []

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        read_calls.append(path)
        return loaded_df

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        write_calls.append((path, index))
        storage[str(path)] = "written"

    monkeypatch.setattr(bgid.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    store = bgid.FileParquetStore(cast(Path, parquet_path))
    parquet_path.write_text("existing")

    assert store.exists()
    assert store.load() is loaded_df

    store.save(pd.DataFrame({"id": [2]}))
    assert write_calls == [(parquet_path, False)]
    assert "written" in storage.values()


def test_resolve_parquet_store_returns_custom() -> None:
    store = InMemoryParquetStore()
    result = bgid._resolve_parquet_store(
        None, store
    )  # pyright: ignore[reportPrivateUsage]
    assert result is store


def test_resolve_parquet_store_with_path(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    def fake_store(path: Path) -> object:
        return sentinel

    monkeypatch.setattr(bgid, "FileParquetStore", fake_store)
    store = bgid._resolve_parquet_store(
        Path("file"), None
    )  # pyright: ignore[reportPrivateUsage]
    assert store is sentinel


def test_resolve_parquet_store_requires_path() -> None:
    with pytest.raises(ValueError):
        bgid._resolve_parquet_store(None, None)  # pyright: ignore[reportPrivateUsage]


def test_load_and_save_checkpoint_roundtrip(root_path: InMemoryPath) -> None:
    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    checkpoint.write_text('{"last_page": 2}')
    checkpoint_path = cast(Path, checkpoint)
    assert bgid.load_checkpoint(checkpoint_path) == 2

    bgid.save_checkpoint(checkpoint_path, 5)
    assert json.loads(checkpoint.storage[checkpoint.path])["last_page"] == 5


def test_load_checkpoint_missing_returns_zero(root_path: InMemoryPath) -> None:
    checkpoint: InMemoryPath = root_path / "missing.json"
    assert bgid.load_checkpoint(cast(Path, checkpoint)) == 0


def test_load_checkpoint_invalid_json_returns_zero(root_path: InMemoryPath) -> None:
    checkpoint: InMemoryPath = root_path / "bad.json"
    checkpoint.write_text("{not-json}")

    assert bgid.load_checkpoint(cast(Path, checkpoint)) == 0


def test_fetch_books_incremental_requires_language(
    root_path: InMemoryPath, in_memory_parquet_store: InMemoryParquetStore
) -> None:
    with pytest.raises(ValueError):
        bgid.fetch_books_incremental(
            languages=[],
            english_only=True,
            checkpoint_path=cast(Path, root_path / "checkpoint.json"),
            parquet_path=cast(Path, root_path / "parquet.parquet"),
            parquet_store=in_memory_parquet_store,
        )


def test_fetch_books_incremental_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    responses = [
        make_response(
            json_payload={
                "results": [
                    {
                        "id": 1,
                        "title": "Book A",
                        "authors": [{"name": "Author X"}],
                        "subjects": ["Fiction"],
                        "bookshelves": ["Adventure"],
                        "languages": ["en"],
                        "download_count": 10,
                        "media_type": "Text",
                        "copyright": False,
                    }
                ],
                "next": "page2",
            }
        ),
        make_response(
            json_payload={
                "results": [
                    {
                        "id": 2,
                        "title": "Book B",
                        "authors": [{"name": "Author Y"}],
                        "subjects": ["Poetry"],
                        "bookshelves": ["Poems"],
                        "languages": ["en"],
                        "download_count": 5,
                        "media_type": "Text",
                        "copyright": False,
                    }
                ],
                "next": None,
            }
        ),
    ]

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    parquet_path = InMemoryPath("parquet", root_path.storage)
    df = bgid.fetch_books_incremental(
        languages=["en"],
        english_only=True,
        checkpoint_path=cast(Path, checkpoint),
        parquet_path=cast(Path, parquet_path),
        parquet_store=in_memory_parquet_store,
    )

    ids = cast(list[int], df["id"].tolist())  # pyright: ignore[reportUnknownMemberType]
    assert set(ids) == {1, 2}
    assert in_memory_parquet_store.exists()
    assert checkpoint.exists()


def test_fetch_books_incremental_handles_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    rate_limit_response = requests.Response()
    rate_limit_response.status_code = 429
    responses = [
        requests.HTTPError(response=rate_limit_response),
        make_response(json_payload={"results": [], "next": None}),
    ]
    attempts = {"count": 0}

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise responses[0]
        return responses[1]

    sleep_calls: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(bgid.time, "sleep", fake_sleep)

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    parquet_path = InMemoryPath("parquet", root_path.storage)
    df = bgid.fetch_books_incremental(
        languages=["en"],
        english_only=True,
        checkpoint_path=cast(Path, checkpoint),
        parquet_path=cast(Path, parquet_path),
        parquet_store=in_memory_parquet_store,
    )

    assert isinstance(df, pd.DataFrame)
    assert sleep_calls  # retry happened


def test_fetch_books_incremental_rate_limit_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    rate_limit_response = requests.Response()
    rate_limit_response.status_code = 429
    attempts = {"count": 0}
    sleep_delays: list[float] = []

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        attempts["count"] += 1
        raise requests.HTTPError(response=rate_limit_response)

    def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(bgid.time, "sleep", fake_sleep)

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    parquet_path = InMemoryPath("parquet", root_path.storage)
    df = bgid.fetch_books_incremental(
        languages=["en"],
        english_only=True,
        checkpoint_path=cast(Path, checkpoint),
        parquet_path=cast(Path, parquet_path),
        parquet_store=in_memory_parquet_store,
    )

    assert df.empty
    assert attempts["count"] == bgid.MAX_RETRIES
    assert sleep_delays == [bgid.INITIAL_RETRY_DELAY, 4.0, 8.0, 16.0]


def test_fetch_books_incremental_raises_for_other_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    response = requests.Response()
    response.status_code = 500

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(requests, "get", fake_get)

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    parquet_path = InMemoryPath("parquet", root_path.storage)

    with pytest.raises(requests.HTTPError):
        bgid.fetch_books_incremental(
            languages=["en"],
            english_only=True,
            checkpoint_path=cast(Path, checkpoint),
            parquet_path=cast(Path, parquet_path),
            parquet_store=in_memory_parquet_store,
        )


def test_fetch_books_incremental_resumes_and_filters_multi_language(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    in_memory_parquet_store.save(
        pd.DataFrame(
            {
                "id": [99],
                "title": ["Existing"],
                "authors": ["Someone"],
                "subjects": ["Existing"],
                "bookshelves": ["Shelf"],
                "languages": ["en"],
                "download_count": [1],
                "media_type": ["Text"],
                "copyright": [False],
            }
        )
    )

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    checkpoint.write_text('{"last_page": 1}')
    parquet_path = InMemoryPath("parquet", root_path.storage)

    responses = [
        make_response(json_payload={"results": [], "next": "page2"}),
        make_response(
            json_payload={
                "results": [
                    {"id": 1, "languages": ["en", "fr"]},
                    {"id": 2, "languages": ["en"]},
                ],
                "next": None,
            }
        ),
    ]

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)

    df = bgid.fetch_books_incremental(
        languages=["en"],
        english_only=True,
        checkpoint_path=cast(Path, checkpoint),
        parquet_path=cast(Path, parquet_path),
        parquet_store=in_memory_parquet_store,
    )

    ids = cast(list[int], df["id"].tolist())  # pyright: ignore[reportUnknownMemberType]
    assert ids == [99, 2]
    assert in_memory_parquet_store.exists()
    assert checkpoint.exists()


def test_fetch_books_incremental_logs_every_tenth_page(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_parquet_store: InMemoryParquetStore,
    root_path: InMemoryPath,
) -> None:
    responses = [
        make_response(json_payload={"results": [], "next": "page2"}),
        make_response(json_payload={"results": [], "next": "page3"}),
        make_response(json_payload={"results": [], "next": "page4"}),
        make_response(json_payload={"results": [], "next": "page5"}),
        make_response(json_payload={"results": [], "next": "page6"}),
        make_response(json_payload={"results": [], "next": "page7"}),
        make_response(json_payload={"results": [], "next": "page8"}),
        make_response(json_payload={"results": [], "next": "page9"}),
        make_response(json_payload={"results": [], "next": "page10"}),
        make_response(json_payload={"results": [], "next": None}),
    ]

    def fake_get(url: str, params: Any | None = None, timeout: int = 60) -> Any:
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)

    checkpoint: InMemoryPath = root_path / "checkpoint.json"
    parquet_path = InMemoryPath("parquet", root_path.storage)

    df = bgid.fetch_books_incremental(
        languages=["en"],
        english_only=True,
        checkpoint_path=cast(Path, checkpoint),
        parquet_path=cast(Path, parquet_path),
        parquet_store=in_memory_parquet_store,
    )

    assert df.empty
    assert not responses
    assert checkpoint.exists()


def test_main_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, root_path: InMemoryPath
) -> None:
    output_path = cast(Path, root_path / "ids.txt")
    parquet_path = cast(Path, root_path / "books.parquet")
    checkpoint_path = cast(Path, root_path / "checkpoint.json")

    sample_df = pd.DataFrame({"id": [3, 1, 2]})

    def fake_fetch_books_incremental(*args: object, **kwargs: object) -> pd.DataFrame:
        return sample_df

    monkeypatch.setattr(bgid, "fetch_books_incremental", fake_fetch_books_incremental)

    def fake_parse_args(self: object) -> SimpleNamespace:
        return SimpleNamespace(
            output=output_path,
            parquet=parquet_path,
            checkpoint=checkpoint_path,
            languages=["en"],
            allow_multi_language=False,
        )

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", fake_parse_args)

    checkpoint_path.write_text("{}")

    bgid.main()

    output_text = output_path.read_text()
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    assert lines[0].startswith("# Autogenerated from Gutendex")
    assert lines[1:] == ["1", "2", "3"]
    assert not checkpoint_path.exists()

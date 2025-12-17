from __future__ import annotations

# ruff: noqa: TC006  # allow explicit typing casts for test doubles
# pyright: reportUnknownMemberType=false
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
import requests

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class InMemoryPath:
    """Lightweight in-memory Path stand-in to avoid filesystem I/O."""

    path: str
    storage: dict[str, str]

    def __truediv__(self, other: str) -> InMemoryPath:
        new_path = f"{self.path}/{other}" if self.path else other
        return InMemoryPath(new_path, self.storage)

    @property
    def parent(self) -> InMemoryPath:
        if "/" not in self.path:
            return self
        parent_path = self.path.rsplit("/", 1)[0]
        return InMemoryPath(parent_path, self.storage)

    def exists(self) -> bool:
        return self.path in self.storage

    def mkdir(
        self, parents: bool = False, exist_ok: bool = False
    ) -> None:  # noqa: ARG002 - match Path API
        return None

    def unlink(self) -> None:
        self.storage.pop(self.path, None)

    def open(
        self, mode: str = "r", encoding: str | None = "utf-8"
    ):  # noqa: ARG002 - match Path API
        if "r" in mode:
            if not self.exists():
                raise FileNotFoundError(self.path)
            stream = io.StringIO(self.storage[self.path])

            class Reader:
                def __enter__(self) -> io.StringIO:
                    return stream

                def __exit__(
                    self, exc_type: Any, exc: Any, tb: Any
                ) -> None:  # noqa: ANN401 - context manager protocol
                    stream.close()

            return Reader()

        stream = io.StringIO()
        storage = self.storage
        self_path = self.path

        class Writer:
            def __enter__(self) -> io.StringIO:
                return stream

            def __exit__(
                self, exc_type: Any, exc: Any, tb: Any
            ) -> None:  # noqa: ANN401 - context manager protocol
                storage[self_path] = stream.getvalue()
                stream.close()

        return Writer()

    def write_text(
        self, data: str, encoding: str = "utf-8"
    ) -> None:  # noqa: ARG002 - match Path API
        self.storage[self.path] = data

    def read_text(
        self, encoding: str = "utf-8"
    ) -> str:  # noqa: ARG002 - match Path API
        if not self.exists():
            raise FileNotFoundError(self.path)
        return self.storage[self.path]

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"InMemoryPath({self.path})"


class InMemoryParquetStore:
    """Simple in-memory Parquet store that mirrors ParquetStore protocol."""

    def __init__(self) -> None:
        self._data: dict[str, list[Any]] | None = None

    def exists(self) -> bool:
        return self._data is not None

    def load(self) -> pd.DataFrame:
        if self._data is None:
            raise FileNotFoundError("No parquet data stored")
        return pd.DataFrame(self._data)

    def save(self, df: pd.DataFrame) -> None:
        raw_dict: dict[str, list[Any]] = cast(
            dict[str, list[Any]], df.to_dict(orient="list")
        )  # pyright: ignore[reportUnknownMemberType]
        data: dict[str, list[Any]] = {
            str(key): list(value) for key, value in raw_dict.items()
        }
        self._data = data


@pytest.fixture()
def path_storage() -> dict[str, str]:
    """Shared storage dict for in-memory paths per test."""

    return {}


@pytest.fixture()
def root_path(path_storage: dict[str, str]) -> InMemoryPath:
    return InMemoryPath("root", path_storage)


@pytest.fixture()
def in_memory_parquet_store() -> InMemoryParquetStore:
    return InMemoryParquetStore()


def make_response(
    *,
    json_payload: dict[str, Any],
    status_code: int = 200,
    raise_for_status: Callable[[], None] | None = None,
) -> Any:
    """Factory for a minimal requests.Response-like object."""

    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code

        def json(self) -> dict[str, Any]:
            return json_payload

        def raise_for_status(self) -> None:
            if raise_for_status:
                raise_for_status()
            if status_code >= 400:
                response = requests.Response()
                response.status_code = status_code
                raise requests.HTTPError(response=response)

    return _Response()

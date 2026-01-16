"""Header forwarding tests for corpus download helpers."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from lexile_corpus_tuner.lexile_scoring_model.corpus import download

if TYPE_CHECKING:
    import pytest


def test_download_file_forwards_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `_download_file` forwards headers to `requests.get` without disk I/O.

    Purpose:
        Validate header propagation while keeping downloads in memory to avoid
        temporary files.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to replace I/O and HTTP calls.

    Returns:
        None: Assertions confirm header forwarding and chunk handling.

    Side Effects:
        Temporarily patches `Path.open`, `Path.replace`, and `requests.get`.
    """

    buffer_store: dict[str, io.BytesIO] = {}
    captured: dict[str, Any] = {}

    def fake_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        """
        Return an in-memory binary buffer in place of filesystem writes.

        Purpose:
            Keep download writes in memory so the test does not create temporary files.

        Args:
            self (Path): The target path being opened.
            mode (str): File mode; must include binary writes for downloads.

        Returns:
            contextlib.AbstractContextManager[io.BytesIO]: Buffer for writing chunks.

        Side Effects:
            Resets any existing buffer for the provided path.
        """
        if "b" not in mode:
            raise AssertionError("Binary mode expected for downloads.")
        buffer = buffer_store.setdefault(str(self), io.BytesIO())
        buffer.seek(0)
        buffer.truncate(0)
        return contextlib.nullcontext(buffer)

    def fake_replace(self: Path, target: Path) -> Path:
        """
        Copy buffered content to the destination placeholder without filesystem writes.

        Purpose:
            Mirror the semantics of `Path.replace` while staying in-memory.

        Args:
            self (Path): Source temporary path.
            target (Path): Destination path for the buffered payload.

        Returns:
            Path: The provided destination path.

        Side Effects:
            Updates the destination buffer with the source contents.
        """
        source_buffer = buffer_store.get(str(self), io.BytesIO())
        dest_buffer = buffer_store.setdefault(str(target), io.BytesIO())
        dest_buffer.seek(0)
        dest_buffer.truncate(0)
        dest_buffer.write(source_buffer.getvalue())
        return target

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
        headers: dict[str, str] | None,
    ) -> MagicMock:
        """
        Capture request parameters and provide a minimal streaming response.

        Purpose:
            Assert that `_download_file` forwards headers and uses streaming semantics.

        Args:
            url (str): URL passed into `_download_file`.
            stream (bool): Stream flag forwarded to `requests.get`.
            timeout (int): Timeout forwarded to `requests.get`.
            headers (dict[str, str] | None): Headers forwarded to `requests.get`.

        Returns:
            MagicMock: Response mock exposing `iter_content` and context manager
                methods.
        """
        captured.update(
            {"url": url, "stream": stream, "timeout": timeout, "headers": headers}
        )
        response = MagicMock()
        response.iter_content.return_value = [b"payload"]
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(Path, "replace", fake_replace)
    monkeypatch.setattr("requests.get", fake_get)

    headers = {
        "User-Agent": "pytest",
        "Referer": "https://example.test/",
    }

    dest = Path("ck12-section.json")
    download_fn = download._download_file  # pyright: ignore[reportPrivateUsage]
    download_fn("https://example.test/resource", dest, headers=headers)

    assert captured["headers"] == headers
    assert buffer_store[str(dest)].getvalue() == b"payload"

"""Header forwarding tests for corpus download helpers."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from lexile_corpus_tuner.lexile_scoring_model.corpus import download

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    assert captured["url"] == "https://example.test/resource"
    assert captured["stream"] is True
    assert captured["timeout"] == 60
    assert captured["headers"] == headers
    assert buffer_store[str(dest)].getvalue() == b"payload"


def test_download_oer_sources_uses_ck12_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure CK-12 manifest entries are downloaded with browser-like headers.

    Purpose:
        Validate that `download_oer_sources` injects the required CK-12 headers
        while leaving other sources untouched.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to replace I/O and HTTP calls.

    Returns:
        None: Assertions confirm header routing per source.

    Side Effects:
        Temporarily patches manifest access, directory creation, and download calls.
    """

    ck12_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.ck12.org/",
        "Origin": "https://www.ck12.org",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    manifest_payload = {
        "sources": [
            {
                "source_id": "openstax",
                "id": "openstax-sample",
                "url": "https://example.test/openstax.txt",
                "filename": "openstax.txt",
            },
            {
                "source_id": "ck12",
                "id": "ck12-section",
                "url": "https://example.test/ck12.json",
                "filename": "ck12.json",
            },
        ]
    }

    class FakeManifest:
        """In-memory manifest path replacement to avoid filesystem I/O."""

        def __init__(self, text: str) -> None:
            self._text = text

        def exists(self) -> bool:
            """Always report the manifest as present."""
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            """Return the JSON manifest payload."""
            return self._text

        def __str__(self) -> str:
            """Provide a readable path representation for logging."""
            return "fake-manifest.json"

    captured_calls: list[dict[str, Any]] = []

    def fake_download_file(
        url: str,
        dest: Path,
        chunk_size: int = 1 << 14,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """
        Capture download invocations without performing network or disk I/O.

        Purpose:
            Track per-source header propagation to confirm CK-12 routing logic.

        Args:
            url (str): Download URL provided by the manifest.
            dest (Path): Destination path computed by the downloader.
            chunk_size (int): Unused placeholder to mirror `_download_file`.
            headers (Mapping[str, str] | None): Headers forwarded by the downloader.
        """

        # Record each download call for later assertions.
        captured_calls.append(
            {
                "url": url,
                "dest": dest,
                "chunk_size": chunk_size,
                "headers": headers,
            }
        )

    manifest = FakeManifest(json.dumps(manifest_payload))
    monkeypatch.setattr(download, "OER_MANIFEST", manifest)
    monkeypatch.setattr(download, "ensure_dirs", lambda: None)
    monkeypatch.setattr(download, "_download_file", fake_download_file)

    download.download_oer_sources()

    # Locate captured calls by destination folder to verify header usage per source.
    ck12_call = next(
        call for call in captured_calls if call["dest"].parent.name == "ck12"
    )
    openstax_call = next(
        call for call in captured_calls if call["dest"].parent.name == "openstax"
    )

    assert ck12_call["headers"] == ck12_headers
    assert openstax_call["headers"] is None

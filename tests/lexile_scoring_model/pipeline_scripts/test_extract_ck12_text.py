"""
Tests for CK-12 revision JSON XHTML extraction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import (
    extract_ck12_text,
)

if TYPE_CHECKING:
    from pytest import LogCaptureFixture

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_extract_uses_primary_xhtml_when_available() -> None:
    """
    Extractor should favor `xhtml` and emit non-empty text when both XHTML fields exist.
    """
    fixture_path = FIXTURE_DIR / "ck12_revision_with_xhtml.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Ensure the fallback XHTML contains distinct text so selection is observable.
    payload["response"]["lesson"][
        "xhtml_prime"
    ] = "<div><p>Prime-only content that should not be used.</p></div>"

    extracted_text = extract_ck12_text.extract_text_from_revision_payload(payload)

    assert extracted_text is not None
    assert "Newtonian Motion" in extracted_text
    assert "Prime-only content" not in extracted_text
    assert extracted_text.strip(), "Extraction should yield non-empty text"


def test_extract_falls_back_to_xhtml_prime_when_primary_missing() -> None:
    """
    Extractor should use `xhtml_prime` when `xhtml` is absent.
    """
    fixture_path = FIXTURE_DIR / "ck12_revision_with_xhtml.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    payload["response"]["lesson"].pop("xhtml", None)
    payload["response"]["lesson"][
        "xhtml_prime"
    ] = "<div><p>Prime fallback content that must be extracted.</p></div>"

    extracted_text = extract_ck12_text.extract_text_from_revision_payload(payload)

    assert extracted_text is not None
    assert "Prime fallback content that must be extracted." in extracted_text
    assert extracted_text.strip(), "Extraction should yield non-empty text"


def test_extract_logs_and_skips_when_xhtml_missing(
    caplog: LogCaptureFixture,
) -> None:
    """
    Extractor should log and skip when neither XHTML field is available.
    """
    payload = {"response": {"lesson": {"title": "No XHTML content"}}}

    with caplog.at_level("ERROR"):
        extracted_text = extract_ck12_text.extract_text_from_revision_payload(payload)

    assert extracted_text is None
    assert "Missing CK-12 lesson XHTML content." in caplog.text


def test_process_directory_emits_outputs_and_warnings(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """
    Directory processor should write parallel .txt files and warn on short text.
    """
    input_dir = Path("/fake/input")
    output_dir = Path("/fake/output")
    json_paths = [
        input_dir / "section-1.json",
        input_dir / "section-2.json",
    ]

    # Normalize path checks to operate entirely in-memory.
    def _always_exists(path: Path) -> bool:
        return True

    def _is_dir(path: Path) -> bool:
        return path in {input_dir, output_dir}

    def _iterdir(path: Path) -> list[Path]:
        return list(json_paths if path == input_dir else [])

    def _is_file(path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def _noop_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(Path, "exists", _always_exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)
    monkeypatch.setattr(Path, "iterdir", _iterdir)
    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(Path, "mkdir", _noop_mkdir)

    payloads: dict[Path, dict[str, str]] = {
        json_paths[0]: {"text": "short text"},
        json_paths[1]: {"text": "long text " * 20},
    }

    def _load_payload(path: Path) -> dict[str, str]:
        return payloads[path]

    def _extract_text(payload: dict[str, str]) -> str:
        return payload["text"]

    monkeypatch.setattr(extract_ck12_text, "_load_revision_payload", _load_payload)
    monkeypatch.setattr(
        extract_ck12_text,
        "extract_text_from_revision_payload",
        _extract_text,
    )

    captured_outputs: dict[Path, str] = {}

    def _capture_write(text: str, output_path: Path) -> None:
        """Capture write calls without touching disk."""
        captured_outputs[output_path] = text

    monkeypatch.setattr(extract_ck12_text, "_write_text_file", _capture_write)

    with caplog.at_level(logging.WARNING, logger=extract_ck12_text.logger.name):
        extract_ck12_text.process_ck12_revision_directory(input_dir, output_dir)

    assert captured_outputs == {
        output_dir / "section-1.txt": "short text",
        output_dir / "section-2.txt": "long text " * 20,
    }
    assert "under 100 chars" in caplog.text


def test_process_directory_logs_failures_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """
    Directory processor should log failures and emit a summary warning.
    """
    input_dir = Path("/fake/input")
    output_dir = Path("/fake/output")
    json_path = input_dir / "broken.json"

    def _always_exists(path: Path) -> bool:
        return True

    def _is_dir(path: Path) -> bool:
        return path in {input_dir, output_dir}

    def _iterdir(path: Path) -> list[Path]:
        return [json_path] if path == input_dir else []

    def _is_file(path: Path) -> bool:
        return path == json_path

    def _noop_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(Path, "exists", _always_exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)
    monkeypatch.setattr(Path, "iterdir", _iterdir)
    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(Path, "mkdir", _noop_mkdir)

    def _load_payload(path: Path) -> dict[str, str]:
        return {"text": "ignored"} if path == json_path else {}

    def _extract_none(_: dict[str, str]) -> None:
        return None

    def _noop_write(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(extract_ck12_text, "_load_revision_payload", _load_payload)
    monkeypatch.setattr(
        extract_ck12_text,
        "extract_text_from_revision_payload",
        _extract_none,
    )
    monkeypatch.setattr(extract_ck12_text, "_write_text_file", _noop_write)

    with caplog.at_level(logging.WARNING, logger=extract_ck12_text.logger.name):
        extract_ck12_text.process_ck12_revision_directory(input_dir, output_dir)

    assert "Failed to extract CK-12 JSON" in caplog.text
    assert "completed with 1 failures" in caplog.text


def test_cli_rejects_non_ck12_source() -> None:
    """CLI should guard against unsupported sources before processing."""
    with pytest.raises(ValueError, match="Unsupported source"):
        extract_ck12_text.extract_ck12_text(source="openstax")


def test_extract_raises_when_xhtml_yields_empty_text(
    caplog: LogCaptureFixture,
) -> None:
    """
    Extractor should return None when XHTML is present but contains no text content.

    Tests line 106: the ValueError raised by _convert_xhtml_to_text when text is empty.
    """
    # Payload has XHTML but it produces empty text after stripping.
    payload = {"response": {"lesson": {"xhtml": "<div>  </div>"}}}

    with caplog.at_level("ERROR"):
        extracted_text = extract_ck12_text.extract_text_from_revision_payload(payload)

    assert extracted_text is None
    assert "No text extracted" in caplog.text


def test_load_revision_payload_raises_for_missing_file() -> None:
    """
    _load_revision_payload should raise FileNotFoundError when the file does not exist.

    Tests lines 165-167: file existence check and error.
    """
    fake_path = Path("/fake/does_not_exist.json")

    with pytest.raises(FileNotFoundError, match="Revision JSON not found"):
        extract_ck12_text._load_revision_payload(  # pyright: ignore[reportPrivateUsage]
            fake_path
        )


def test_load_revision_payload_parses_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _load_revision_payload should return the parsed JSON when the file is valid.

    Tests line 168: the json.loads call.
    """
    expected = {"response": {"lesson": {"xhtml": "<p>Test</p>"}}}

    def _is_file(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return True

    def _read_text(
        self: Path, encoding: str = "utf-8"  # noqa: ARG001 - monkeypatch signature
    ) -> str:
        return json.dumps(expected)

    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(Path, "read_text", _read_text)

    result = (
        extract_ck12_text._load_revision_payload(  # pyright: ignore[reportPrivateUsage]
            Path("/fake/valid.json")
        )
    )
    assert result == expected


def test_write_text_file_creates_parent_dirs_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _write_text_file should create parent directories and write the text atomically.

    Tests lines 193-200: validation, mkdir, temp write, and replace.
    """
    mkdir_called = False
    write_text_calls: list[str] = []
    replace_called = False

    class MockPath:
        """In-memory path stub for write testing."""

        def __init__(self, path_str: str) -> None:
            self._path = path_str

        @property
        def name(self) -> str:
            return self._path.rsplit("/", 1)[-1] if "/" in self._path else self._path

        @property
        def parent(self) -> MockPath:
            parent_str = self._path.rsplit("/", 1)[0] if "/" in self._path else ""
            return MockPath(parent_str)

        @property
        def suffix(self) -> str:
            return "." + self.name.split(".")[-1] if "." in self.name else ""

        def with_suffix(self, suffix: str) -> MockPath:
            return MockPath(self._path + suffix)

        def mkdir(
            self,
            parents: bool = False,  # noqa: ARG002 - mock API signature
            exist_ok: bool = False,  # noqa: ARG002 - mock API signature
        ) -> None:
            nonlocal mkdir_called
            mkdir_called = True

        def write_text(
            self,
            text: str,
            encoding: str = "utf-8",  # noqa: ARG002 - mock API signature
        ) -> None:
            write_text_calls.append(text)

        def replace(
            self, target: MockPath
        ) -> None:  # noqa: ARG002 - mock API signature
            nonlocal replace_called
            replace_called = True

    output_path = MockPath("/fake/out/file.txt")
    extract_ck12_text._write_text_file(  # pyright: ignore[reportPrivateUsage]
        "Sample text", output_path  # type: ignore[arg-type]
    )

    assert mkdir_called
    assert len(write_text_calls) == 1
    assert write_text_calls[0] == "Sample text"
    assert replace_called


def test_write_text_file_rejects_path_without_filename() -> None:
    """
    _write_text_file should raise ValueError when the output path has no filename.

    Tests lines 193-194: the filename validation.
    """

    class EmptyNamePath:
        """Path with empty name for testing validation."""

        name = ""

    with pytest.raises(ValueError, match="Output path must include a filename"):
        extract_ck12_text._write_text_file(  # pyright: ignore[reportPrivateUsage]
            "text", EmptyNamePath()  # type: ignore[arg-type]
        )


def test_process_directory_raises_for_missing_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    process_ck12_revision_directory should raise FileNotFoundError for missing input.

    Tests line 231: input directory existence check.
    """

    def _exists(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return False

    monkeypatch.setattr(Path, "exists", _exists)

    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        extract_ck12_text.process_ck12_revision_directory(
            Path("/fake/missing"), Path("/fake/output")
        )


def test_process_directory_raises_for_non_directory_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    process_ck12_revision_directory should raise NotADirectoryError for file input.

    Tests line 234: input path directory check.
    """

    def _exists(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return True

    def _is_dir(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return False

    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)

    with pytest.raises(NotADirectoryError, match="Input path is not a directory"):
        extract_ck12_text.process_ck12_revision_directory(
            Path("/fake/file.txt"), Path("/fake/output")
        )


def test_process_directory_handles_empty_input_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    process_ck12_revision_directory should return early with no JSON files.

    Tests line 246: early return when directory has no JSON files.
    """
    mkdir_calls: list[Path] = []

    def _exists(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return True

    def _is_dir(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return True

    def _iterdir(self: Path) -> list[Path]:  # noqa: ARG001 - monkeypatch signature
        return []

    def _capture_mkdir(
        self: Path,
        parents: bool = False,  # noqa: ARG001 - monkeypatch signature
        exist_ok: bool = False,  # noqa: ARG001 - monkeypatch signature
    ) -> None:
        mkdir_calls.append(self)

    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)
    monkeypatch.setattr(Path, "iterdir", _iterdir)
    monkeypatch.setattr(Path, "mkdir", _capture_mkdir)

    # Should not raise, just return early.
    extract_ck12_text.process_ck12_revision_directory(
        Path("/fake/empty_dir"), Path("/fake/output")
    )

    # mkdir for output_dir should still be called before the early return.
    assert len(mkdir_calls) == 1


def test_process_directory_handles_timeout_gracefully(
    monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """
    Directory processor should handle timeouts per file without halting the batch.

    Tests lines 294-296: timeout exception handling in the futures loop.
    """
    import concurrent.futures

    input_dir = Path("/fake/input")
    output_dir = Path("/fake/output")
    json_path = input_dir / "slow.json"

    def _exists(self: Path) -> bool:  # noqa: ARG001 - monkeypatch signature
        return True

    def _is_dir(self: Path) -> bool:
        return self in {input_dir, output_dir}

    def _iterdir(self: Path) -> list[Path]:
        return [json_path] if self == input_dir else []

    def _is_file(self: Path) -> bool:
        return self == json_path

    def _mkdir(
        self: Path,  # noqa: ARG001 - monkeypatch signature
        parents: bool = False,  # noqa: ARG001 - monkeypatch signature
        exist_ok: bool = False,  # noqa: ARG001 - monkeypatch signature
    ) -> None:
        pass

    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)
    monkeypatch.setattr(Path, "iterdir", _iterdir)
    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(Path, "mkdir", _mkdir)

    class TimeoutFuture:
        """Future that always times out."""

        def result(
            self,
            timeout: float = 0,  # noqa: ARG002 - mock API signature
        ) -> None:
            raise concurrent.futures.TimeoutError("simulated timeout")

        def cancel(self) -> bool:
            return True

    # Patch at module level to intercept the ThreadPoolExecutor usage.
    def _patched_executor(
        max_workers: int,  # noqa: ARG001 - mock API signature
    ) -> object:
        class PatchedExecutor:
            def __enter__(self) -> PatchedExecutor:
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def submit(
                self,
                fn: object,  # noqa: ARG002 - mock API signature
                *args: object,
                **kwargs: object,
            ) -> TimeoutFuture:
                return TimeoutFuture()

        return PatchedExecutor()

    monkeypatch.setattr(
        concurrent.futures, "ThreadPoolExecutor", _patched_executor  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.WARNING, logger=extract_ck12_text.logger.name):
        extract_ck12_text.process_ck12_revision_directory(input_dir, output_dir)

    # Timeout is logged at ERROR level, summary at WARNING level.
    assert "Timed out extracting CK-12 JSON" in caplog.text
    assert "completed with 1 failures" in caplog.text


def test_cli_happy_path_invokes_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CLI entry point should invoke the directory processor for valid CK-12 source.

    Tests lines 361-366: the typer.echo and process_ck12_revision_directory calls.
    """
    calls: list[tuple[Path, Path]] = []

    def _capture_process(input_dir: Path, output_dir: Path) -> None:
        calls.append((input_dir, output_dir))

    monkeypatch.setattr(
        extract_ck12_text, "process_ck12_revision_directory", _capture_process
    )

    # Capture typer.echo output to avoid print noise.
    echoes: list[object] = []

    def _capture_echo(msg: object) -> None:
        echoes.append(msg)

    monkeypatch.setattr("typer.echo", _capture_echo)

    input_path = Path("/fake/input")
    output_path = Path("/fake/output")
    extract_ck12_text.extract_ck12_text(
        source="ck12", input_dir=input_path, output_dir=output_path
    )

    assert len(calls) == 1
    assert calls[0] == (input_path, output_path)
    assert any("Extracting CK-12" in str(e) for e in echoes)
    assert any("complete" in str(e).lower() for e in echoes)

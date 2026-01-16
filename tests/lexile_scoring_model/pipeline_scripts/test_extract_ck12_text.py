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

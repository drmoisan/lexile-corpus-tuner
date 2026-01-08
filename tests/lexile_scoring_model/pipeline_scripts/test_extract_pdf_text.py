from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import extract_pdf_text

if TYPE_CHECKING:
    from types import TracebackType


def test_extract_text_from_pdf_returns_concatenated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    extract_text_from_pdf should combine per-page text when extraction succeeds.

    Purpose:
        Validate that successful pdfplumber extraction concatenates page content
        and returns the aggregated string without trailing whitespace.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to stub filesystem checks and
            pdfplumber.open for deterministic, in-memory behavior.
    """

    class _FakePage:
        """Stub pdfplumber page that returns predetermined text."""

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            """Return the preconfigured page text."""
            return self._text

    class _FakePdf:
        """Context manager stub supplying fake pages to the extractor."""

        def __init__(self, pages: list[_FakePage]) -> None:
            self.pages = pages

        def __enter__(self) -> _FakePdf:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    fake_pages = [
        _FakePage("Page one"),
        _FakePage("Page two"),
    ]

    captured_path: dict[str, Path] = {}

    def _fake_open(path: Path) -> _FakePdf:
        """Capture the provided path and return a fake PDF object."""
        captured_path["path"] = path
        return _FakePdf(fake_pages)

    def _pretend_file_exists(_: Path) -> bool:
        """Force Path.is_file to report true for the supplied fake path."""
        return True

    monkeypatch.setattr(extract_pdf_text.pdfplumber, "open", _fake_open)
    monkeypatch.setattr(Path, "is_file", _pretend_file_exists)

    pdf_path = Path("ck12_sample.pdf")

    result = extract_pdf_text.extract_text_from_pdf(pdf_path)

    assert result == "Page one\nPage two"
    assert captured_path["path"] == pdf_path


def test_extract_text_from_pdf_logs_and_raises_when_no_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    extract_text_from_pdf should log and raise when extraction yields no content.

    Purpose:
        Ensure that empty extraction results surface as actionable errors instead
        of silently returning an empty string.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to stub filesystem checks and
            pdfplumber.open for deterministic behavior.
        caplog (pytest.LogCaptureFixture): Fixture to capture log output.
    """

    class _EmptyPage:
        """Stub pdfplumber page that returns no text."""

        def extract_text(self) -> str | None:
            return None

    class _FakePdf:
        """Context manager stub supplying empty pages to the extractor."""

        def __init__(self, pages: list[_EmptyPage]) -> None:
            self.pages = pages

        def __enter__(self) -> _FakePdf:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    empty_pages = [_EmptyPage()]

    def _fake_open(_: Path) -> _FakePdf:
        """Return a fake PDF object with empty content."""
        return _FakePdf(empty_pages)

    def _pretend_file_exists(_: Path) -> bool:
        """Force Path.is_file to report true for the supplied fake path."""
        return True

    monkeypatch.setattr(extract_pdf_text.pdfplumber, "open", _fake_open)
    monkeypatch.setattr(Path, "is_file", _pretend_file_exists)

    pdf_path = Path("empty.pdf")

    with caplog.at_level(logging.ERROR, logger=extract_pdf_text.LOGGER.name):
        with pytest.raises(ValueError, match="No text extracted"):
            extract_pdf_text.extract_text_from_pdf(pdf_path)

    assert any(
        "No text extracted from PDF" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    )

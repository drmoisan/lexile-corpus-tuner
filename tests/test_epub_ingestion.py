from __future__ import annotations

from pathlib import Path

from lexile_corpus_tuner.epub import extract_text_from_epub
from tests.utils import write_minimal_epub


def test_extract_text_from_epub_reads_spine(tmp_path: Path):
    """extract_text_from_epub concatenates XHTML chapters following the spine order."""
    epub_path = tmp_path / "book.epub"
    write_minimal_epub(
        epub_path,
        chapters=[
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Hello crew.</p></body></html>",
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Second chapter.</p></body></html>",
        ],
    )
    text = extract_text_from_epub(epub_path)
    assert "Hello crew." in text
    assert "Second chapter." in text
    assert "Hello crew.\n\nSecond chapter." in text


def test_extract_text_from_epub_falls_back_without_spine(tmp_path: Path):
    """extract_text_from_epub still returns text when the OPF lacks a spine section."""
    epub_path = tmp_path / "fallback.epub"
    write_minimal_epub(
        epub_path,
        chapters=[
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Fallback only.</p></body></html>"
        ],
        include_spine=False,
    )
    text = extract_text_from_epub(epub_path)
    assert "Fallback only." in text

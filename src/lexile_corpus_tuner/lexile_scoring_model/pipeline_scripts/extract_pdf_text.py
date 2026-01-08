"""
CK-12 PDF text extraction utilities for converting downloaded PDFs to text.

Purpose:
    Provide a structured, strongly-typed surface for PDF-to-text extraction with
    deterministic output naming and actionable error logging.

Usage:
    The Typer CLI entrypoint `extract_pdf_text` invokes `process_ck12_directory`
    to walk CK-12 PDF inputs, extract text via `extract_text_from_pdf`, and
    persist results using `save_text_file`.

Flow:
    1) Iterate CK-12 PDF files within an input directory.
    2) Extract raw text content for each PDF with timeout protection.
    3) Write extracted text to deterministic output paths alongside PDFs.
    4) Log failures with PDF and destination context to aid remediation.

Invariants / Constraints:
    - Output filenames must be stable and parallel to their PDF counterparts.
    - Extraction should avoid partial writes by using safe write patterns.
    - Processing must remain testable by isolating I/O and computation.

Side Effects:
    Reads and writes filesystem content and logs extraction failures.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber
import typer

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(
    help="Extract text from CK-12 PDF downloads into parallel .txt files."
)

LOGGER = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from a CK-12 PDF file.

    Purpose:
        Convert a single PDF into text so downstream normalization can consume
        plain-text content instead of PDF binaries.

    Args:
        pdf_path (Path): Absolute or relative path to the PDF file to extract.

    Returns:
        str: Extracted text content from the PDF.

    Raises:
        FileNotFoundError: If the provided path does not exist or is not a file.
        TimeoutError: When extraction exceeds the configured timeout budget.
        ValueError: When no text could be extracted from the PDF.
        pdfplumber.pdf.PDFSyntaxError: If the PDF is malformed.

    Side Effects:
        Opens and reads from the filesystem; ensures file handles are closed even
        on timeout or failure.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _extract() -> str:
        """
        Execute pdfplumber extraction within a worker to allow timeout control.

        Returns:
            str: Concatenated text from all pages.
        """
        # Open the PDF with pdfplumber to guarantee closure via context manager.
        with pdfplumber.open(pdf_path) as pdf:
            extracted_pages: list[str] = []
            # Accumulate page-level text to preserve page order in output.
            for page in pdf.pages:
                # Extract text for each page; empty strings are allowed here so
                # we can evaluate overall emptiness after aggregation.
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)
        return "\n".join(extracted_pages).strip()

    extractor: Callable[[], str] = _extract
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extractor)
        try:
            text = future.result(timeout=30)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Timed out extracting PDF: {pdf_path}") from exc

    if not text:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    return text


def save_text_file(text: str, output_path: Path) -> None:
    """
    Persist extracted text to disk with overwrite safety.

    Purpose:
        Provide a dedicated writer for extracted text so file creation, parent
        directory handling, and overwrite rules stay consistent across callers.

    Args:
        text (str): Extracted text content to write.
        output_path (Path): Destination path for the .txt output.

    Returns:
        None

    Raises:
        ValueError: If the output path does not include a filename.
        OSError: When filesystem interactions fail (e.g., permissions, disk).

    Side Effects:
        Creates parent directories as needed and writes the provided content to
        disk, replacing any existing file atomically.
    """
    if not output_path.name:
        raise ValueError("Output path must include a filename.")

    # Ensure the destination directory exists before writing.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the target directory to avoid partial writes,
    # then atomically replace the destination.
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)


def process_ck12_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Process CK-12 PDF files within a directory and emit parallel text outputs.

    Purpose:
        Orchestrate PDF discovery, extraction, and text persistence so a single
        call can produce text artifacts for all CK-12 downloads.

    Args:
        input_dir (Path): Directory containing CK-12 PDF files.
        output_dir (Path): Directory where extracted text files will be written.

    Returns:
        None

    Raises:
        FileNotFoundError: When the input directory does not exist.
        NotADirectoryError: When the input path is not a directory.
        ValueError: When output file naming cannot be derived.
        OSError: When filesystem operations fail.
        pdfplumber.pdf.PDFSyntaxError: If a PDF is malformed.

    Side Effects:
        Reads PDF files from disk, writes extracted text files to disk, and logs
        extraction failures with PDF and destination context.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect CK-12 PDF files in deterministic order to keep outputs stable.
    pdf_paths: list[Path] = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_paths:
        return

    failures: list[Path] = []

    def _process_single_pdf(pdf_path: Path) -> None:
        """
        Extract text from a single PDF and persist it alongside the corpus.

        Args:
            pdf_path (Path): Source PDF path within the input directory.

        Returns:
            None

        Raises:
            ValueError: When text extraction yields an empty result.
            OSError: When writing the extracted text fails.
            pdfplumber.pdf.PDFSyntaxError: If the PDF is malformed.

        Side Effects:
            Reads from and writes to disk for the provided PDF path.
        """
        output_path = output_dir / f"{pdf_path.stem}.txt"
        text = extract_text_from_pdf(pdf_path)
        save_text_file(text, output_path)

    max_workers = min(4, len(pdf_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their source PDFs so failures can be logged with context.
        futures = {
            executor.submit(_process_single_pdf, pdf_path): pdf_path
            for pdf_path in pdf_paths
        }

        # Consume results, logging actionable error details instead of halting.
        for future in concurrent.futures.as_completed(futures):
            pdf_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                output_path = output_dir / f"{pdf_path.stem}.txt"
                LOGGER.error(
                    "Failed to extract PDF %s to %s (%s): %s",
                    pdf_path,
                    output_path,
                    exc.__class__.__name__,
                    exc,
                )
                failures.append(pdf_path)

    if failures:
        LOGGER.warning(
            (
                "CK-12 extraction completed with %d failures; see errors above. "
                "First failure: %s"
            ),
            len(failures),
            failures[0],
        )


@app.command()
def extract_pdf_text(
    source: str = typer.Option(  # noqa: B008 - Typer framework pattern
        "ck12",
        help="Source identifier for logging or future routing.",
    ),
    input_dir: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/corpus/raw/ck12"),
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory containing downloaded CK-12 PDF files.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/corpus/raw/ck12"),
        file_okay=False,
        resolve_path=True,
        help="Directory where extracted text files will be written.",
    ),
) -> None:
    """
    CLI entry point for extracting text from CK-12 PDFs.

    Purpose:
        Provide a single-command interface that will eventually orchestrate
        directory processing for CK-12 PDF text extraction.

    Args:
        source (str): OER source identifier, reserved for future routing needs.
        input_dir (Path): Directory containing CK-12 PDF downloads.
        output_dir (Path): Directory where extracted text files will be stored.

    Returns:
        None

    Raises:
        ValueError: When a source other than CK-12 is requested.

    Side Effects:
        Reads PDF files from disk, writes extracted text files, and logs
        extraction failures without stopping the batch.
    """
    normalized_source = source.strip().lower()
    if normalized_source != "ck12":
        raise ValueError(f"Unsupported source for PDF extraction: {source}")

    typer.echo(
        f"Extracting PDF text for source '{normalized_source}' "
        f"from {input_dir} to {output_dir}..."
    )
    process_ck12_directory(input_dir, output_dir)
    typer.echo("Extraction complete.")


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

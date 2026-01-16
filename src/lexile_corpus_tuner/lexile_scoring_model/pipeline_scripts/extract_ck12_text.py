"""
CK-12 revision JSON to plain-text extraction helpers.

Purpose:
    Provide a focused, testable surface for converting CK-12 revision payloads
    into normalized plain text so downstream corpus stages can consume CK-12
    content without HTML rendering.

Usage:
    Call `extract_text_from_revision_payload` with a loaded revision JSON object
    to obtain normalized text. A CLI wrapper will orchestrate directory
    traversal in a later task.

Flow:
    1) Select the primary XHTML fragment from the revision payload, preferring
       `response.lesson.xhtml` over `response.lesson.xhtml_prime`.
    2) Parse the XHTML with BeautifulSoup using the `lxml` parser.
    3) Normalize extracted text and ensure it is non-empty before returning.

Invariants / Constraints:
    - XHTML content must be present in one of the expected lesson fields.
    - Returned text must be non-empty after normalization.

Side Effects:
    None. This module only parses in-memory payloads.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

import typer
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Extract plain text from CK-12 revision JSON files into parallel .txt outputs."
)


def _select_lesson_xhtml(payload: dict[str, Any]) -> str:
    """
    Choose the preferred XHTML fragment from a CK-12 revision payload.

    Purpose:
        Surface the XHTML document that should be rendered into plain text,
        preferring the primary `xhtml` field while allowing a fallback to
        `xhtml_prime` when needed.

    Args:
        payload (dict[str, Any]): Parsed CK-12 revision JSON object.

    Returns:
        str: XHTML string selected for conversion.

    Raises:
        ValueError: When neither `xhtml` nor `xhtml_prime` is available.

    Side Effects:
        None.
    """
    lesson = payload.get("response", {}).get("lesson", {})

    # Prefer the canonical XHTML content when available to mirror the CK-12
    # reader experience; fall back only when the primary field is absent.
    primary_xhtml = lesson.get("xhtml")
    if primary_xhtml:
        return str(primary_xhtml)

    fallback_xhtml = lesson.get("xhtml_prime")
    if fallback_xhtml:
        return str(fallback_xhtml)

    raise ValueError("Missing CK-12 lesson XHTML content.")


def _convert_xhtml_to_text(xhtml: str) -> str:
    """
    Convert CK-12 lesson XHTML into normalized plain text.

    Purpose:
        Strip markup while preserving readable ordering so downstream pipelines
        ingest consistent text output.

    Args:
        xhtml (str): XHTML fragment extracted from a revision payload.

    Returns:
        str: Plain-text content derived from the XHTML.

    Raises:
        ValueError: When conversion yields empty text.

    Side Effects:
        None.
    """
    # Use the lxml parser to match repository standards for CK-12 XHTML parsing.
    soup = BeautifulSoup(xhtml, "lxml")
    # Extract textual content while keeping whitespace readable for assertions.
    text = soup.get_text(" ", strip=True)
    if not text:
        raise ValueError("No text extracted from CK-12 XHTML payload.")
    return text


def extract_text_from_revision_payload(payload: dict[str, Any]) -> str | None:
    """
    Extract plain text from a CK-12 revision JSON payload.

    Purpose:
        Provide a single entry point for selecting lesson XHTML and rendering it
        to text, ensuring callers receive non-empty, normalized content when
        available while logging and skipping invalid payloads.

    Args:
        payload (dict[str, Any]): Parsed CK-12 revision JSON object.

    Returns:
        str | None: Plain-text representation of the lesson content, or None
            when XHTML is missing or cannot be converted.

    Side Effects:
        Logs extraction failures for observability.
    """
    # Select the best-available XHTML field before conversion to preserve
    # ordering and authoring intent.
    try:
        xhtml = _select_lesson_xhtml(payload)
    except ValueError as exc:
        logger.error("Skipping CK-12 lesson: %s", exc)
        return None

    try:
        return _convert_xhtml_to_text(xhtml)
    except ValueError as exc:
        logger.error("Skipping CK-12 lesson during conversion: %s", exc)
        return None


def _load_revision_payload(json_path: Path) -> dict[str, Any]:
    """
    Load a CK-12 revision JSON file into memory.

    Purpose:
        Provide deterministic JSON loading with clear errors so calling code can
        attribute failures to specific files.

    Args:
        json_path (Path): Path to the CK-12 revision JSON file.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: When the file contents are not valid JSON.

    Side Effects:
        Reads the JSON file from disk.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Revision JSON not found: {json_path}")

    return json.loads(json_path.read_text(encoding="utf-8"))


def _write_text_file(text: str, output_path: Path) -> None:
    """
    Persist extracted text to disk with overwrite safety.

    Purpose:
        Ensure extracted text is written atomically so partially written outputs
        do not appear if an error occurs mid-write.

    Args:
        text (str): Extracted text content to save.
        output_path (Path): Destination for the resulting `.txt` file.

    Returns:
        None

    Raises:
        ValueError: If the destination path does not include a filename.
        OSError: When filesystem operations fail.

    Side Effects:
        Creates parent directories as needed and writes text content to disk.
    """
    if not output_path.name:
        raise ValueError("Output path must include a filename.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)


def process_ck12_revision_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Convert CK-12 revision JSON files to plain-text outputs.

    Purpose:
        Walk a directory of CK-12 revision payloads, extract XHTML-based text,
        and emit parallel `.txt` files while logging failures without halting
        the batch.

    Args:
        input_dir (Path): Directory containing CK-12 revision JSON files.
        output_dir (Path): Directory where extracted `.txt` files are written.

    Returns:
        None

    Raises:
        FileNotFoundError: When the input directory does not exist.
        NotADirectoryError: When the input path is not a directory.
        ValueError: When extraction yields no text for a file.
        json.JSONDecodeError: When a revision JSON file is invalid.
        OSError: When filesystem interactions fail.

    Side Effects:
        Reads JSON files, writes text files, and logs warnings/errors for
        failures while continuing other work items.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidate revision payloads deterministically so outputs stay stable.
    json_paths: list[Path] = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )

    if not json_paths:
        return

    failures: list[Path] = []

    def _process_single_revision(json_path: Path) -> None:
        """
        Extract text from a single CK-12 revision JSON and persist it.

        Args:
            json_path (Path): Source revision JSON path.

        Returns:
            None

        Raises:
            ValueError: When extraction yields no text.
            json.JSONDecodeError: When the JSON cannot be parsed.

        Side Effects:
            Reads from disk and writes the extracted text to disk.
        """
        payload = _load_revision_payload(json_path)
        text = extract_text_from_revision_payload(payload)
        if text is None:
            raise ValueError("CK-12 revision JSON missing extractable XHTML.")

        if len(text) < 100:
            logger.warning(
                "Extracted text under 100 chars (possible image-only section): %s",
                json_path,
            )

        output_path = output_dir / f"{json_path.stem}.txt"
        _write_text_file(text, output_path)

    max_workers = min(4, len(json_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Fan out extraction while preserving per-file context for error reporting.
        futures = {
            executor.submit(_process_single_revision, json_path): json_path
            for json_path in json_paths
        }

        # Resolve each future with a timeout so individual hangs do not block the batch.
        for future, json_path in futures.items():
            try:
                future.result(timeout=30)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                failures.append(json_path)
                logger.error("Timed out extracting CK-12 JSON %s: %s", json_path, exc)
            except Exception as exc:
                failures.append(json_path)
                output_path = output_dir / f"{json_path.stem}.txt"
                logger.error(
                    "Failed to extract CK-12 JSON %s to %s (%s): %s",
                    json_path,
                    output_path,
                    exc.__class__.__name__,
                    exc,
                )

    if failures:
        logger.warning(
            ("CK-12 text extraction completed with %d failures; " "first failure: %s"),
            len(failures),
            failures[0],
        )


@app.command()
def extract_ck12_text(
    source: str = typer.Option(  # noqa: B008 - Typer framework pattern
        "ck12",
        help="Source identifier; only 'ck12' is supported.",
    ),
    input_dir: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/corpus/raw/ck12"),
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory containing CK-12 revision JSON files.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008 - Typer framework pattern
        Path("data/corpus/raw/ck12"),
        file_okay=False,
        resolve_path=True,
        help="Directory where extracted text files will be written.",
    ),
) -> None:
    """
    CLI entry point for converting CK-12 revision JSON payloads to text.

    Purpose:
        Provide a single-command interface for batch extraction so corpus
        downloads can be transformed into text assets without manual steps.

    Args:
        source (str): OER source identifier; only CK-12 is supported.
        input_dir (Path): Directory containing downloaded CK-12 revision JSON.
        output_dir (Path): Directory where extracted `.txt` files are written.

    Returns:
        None

    Raises:
        ValueError: When a non-CK-12 source is requested.

    Side Effects:
        Reads JSON files, writes text files, and logs extraction outcomes.
    """
    normalized_source = source.strip().lower()
    if normalized_source != "ck12":
        raise ValueError(f"Unsupported source for CK-12 extraction: {source}")

    typer.echo(
        f"Extracting CK-12 revision text for source '{normalized_source}' "
        f"from {input_dir} to {output_dir}..."
    )
    process_ck12_revision_directory(input_dir, output_dir)
    typer.echo("Extraction complete.")


__all__ = [
    "extract_ck12_text",
    "extract_text_from_revision_payload",
    "process_ck12_revision_directory",
]


if __name__ == "__main__":
    app()  # pragma: no cover - CLI dispatch

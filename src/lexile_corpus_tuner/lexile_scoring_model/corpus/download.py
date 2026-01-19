from __future__ import annotations

import bz2
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

RAW_ROOT = Path("data/corpus/raw")
GUTENBERG_DIR = RAW_ROOT / "gutenberg"
SIMPLE_WIKI_DIR = RAW_ROOT / "simple_wiki"
GUTENBERG_IDS_FILE = Path("data/meta/gutenberg/gutenberg_ids.txt")
OER_MANIFEST = Path("data/meta/oer_sources.json")
DEFAULT_SIMPLE_WIKI_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)
CK12_BROWSER_HEADERS: dict[str, str] = {
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


LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Ensure expected raw corpus directories exist."""
    GUTENBERG_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLE_WIKI_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_ROOT / "openstax").mkdir(parents=True, exist_ok=True)
    (RAW_ROOT / "ck12").mkdir(parents=True, exist_ok=True)


def download_gutenberg_subset(limit: int | None = None) -> None:
    """
    Download a curated subset of Project Gutenberg texts into RAW_ROOT/'gutenberg'.
    """
    ensure_dirs()
    ebook_ids = list(_iter_gutenberg_ids(limit))
    if not ebook_ids:
        LOGGER.warning(
            "No Gutenberg IDs found at %s; create the file to enable downloads.",
            GUTENBERG_IDS_FILE,
        )
        return

    for ebook_id in ebook_ids:
        dest = GUTENBERG_DIR / f"{ebook_id}.txt"
        if dest.exists():
            LOGGER.info("Skipping Gutenberg %s (already downloaded).", ebook_id)
            continue
        url = _resolve_gutenberg_url(ebook_id)
        if url is None:
            LOGGER.warning("Unable to construct URL for Gutenberg ID %s.", ebook_id)
            continue
        LOGGER.info("Downloading Gutenberg %s from %s", ebook_id, url)
        try:
            _download_file(url, dest)
        except requests.RequestException as exc:
            LOGGER.error("Failed to download Gutenberg %s: %s", ebook_id, exc)


def download_simple_wiki_dump(dump_url: str | None = None) -> Path:
    """
    Download a Simple English Wikipedia XML dump into RAW_ROOT/'simple_wiki'.

    Purpose:
        Ensure both the compressed `.bz2` dump and an extracted `.xml` file are
        present for downstream processing.

    Args:
        dump_url (str | None): Optional override for the dump URL. When omitted,
            the environment variable `LEXILE_SIMPLE_WIKI_DUMP_URL` or the
            default Simple Wiki URL is used.

    Returns:
        Path: The path to the downloaded `.bz2` archive.

    Raises:
        requests.RequestException: If the download fails.
        OSError: If extraction fails to write or replace the XML output.

    Side Effects:
        Creates the raw corpus directory, downloads the dump if needed, and
        writes an extracted XML file alongside the `.bz2` archive.
    """
    ensure_dirs()
    if dump_url is None:
        dump_url = os.environ.get(
            "LEXILE_SIMPLE_WIKI_DUMP_URL", DEFAULT_SIMPLE_WIKI_URL
        )
    filename = dump_url.split("/")[-1] or "simplewiki_dump.xml.bz2"
    bz2_path = SIMPLE_WIKI_DIR / filename
    xml_path = bz2_path.with_suffix(".xml")
    # Branch by artifact state:
    # - If bz2 exists and XML is non-empty, skip extraction.
    # - If bz2 exists but XML is missing/empty, extract.
    # - If bz2 is missing, download then extract.
    if bz2_path.exists():
        if xml_path.exists():
            xml_size = xml_path.stat().st_size
            if xml_size > 0:
                LOGGER.info(
                    "Skipping Simple Wiki extraction for %s because XML exists at %s "
                    "(size=%s).",
                    bz2_path,
                    xml_path,
                    xml_size,
                )
                return bz2_path
        LOGGER.info(
            "Simple Wiki dump already exists at %s; extracting XML to %s.",
            bz2_path,
            xml_path,
        )
        _extract_simple_wiki_bz2(bz2_path)
        return bz2_path

    LOGGER.info("Downloading Simple Wiki dump from %s to %s", dump_url, bz2_path)
    _download_file(dump_url, bz2_path)
    LOGGER.info("Extracting Simple Wiki dump from %s to %s", bz2_path, xml_path)
    _extract_simple_wiki_bz2(bz2_path)
    return bz2_path


def download_oer_sources() -> None:
    """Download OpenStax / CK-12 excerpts defined in the manifest."""
    ensure_dirs()
    if not OER_MANIFEST.exists():
        LOGGER.info(
            "OER manifest missing at %s; skipping OpenStax/CK-12 downloads.",
            OER_MANIFEST,
        )
        return

    try:
        manifest = json.loads(OER_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse %s: %s", OER_MANIFEST, exc)
        return

    sources = manifest.get("sources", [])
    if not sources:
        LOGGER.info("No sources listed in %s; nothing to download.", OER_MANIFEST)
        return

    # Iterate over each manifest entry to place downloads in source-specific folders.
    for entry in sources:
        url = entry.get("url")
        source_id = (entry.get("source_id") or "oer").lower()
        item_id = entry.get("id") or source_id
        if not url:
            LOGGER.warning("Skipping %s: missing URL in manifest.", item_id)
            continue
        filename = entry.get("filename") or f"{item_id}.txt"
        dest_dir = RAW_ROOT / source_id
        dest = dest_dir / filename
        if dest.exists():
            LOGGER.info("Skipping %s (already downloaded).", dest)
            continue
        LOGGER.info("Downloading %s from %s", item_id, url)
        headers = None
        # CK-12 endpoints require browser-like headers for anonymous access;
        # other sources use default request headers.
        if source_id == "ck12":
            headers = CK12_BROWSER_HEADERS
        try:
            if url.startswith("file://"):
                _copy_local_file(Path(url[7:]), dest)
            elif Path(url).is_absolute():
                _copy_local_file(Path(url), dest)
            else:
                _download_file(url, dest, headers=headers)
        except (requests.RequestException, OSError) as exc:
            LOGGER.error("Failed to download %s: %s", url, exc)


def _iter_gutenberg_ids(limit: int | None) -> Iterator[int]:
    if not GUTENBERG_IDS_FILE.exists():
        return
    count = 0
    with GUTENBERG_IDS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ebook_id = int(line)
            except ValueError:
                continue
            yield ebook_id
            count += 1
            if limit is not None and count >= limit:
                break


def _resolve_gutenberg_url(ebook_id: int) -> str | None:
    # Try a handful of known patterns as Gutenberg filenames vary.
    candidate_patterns = [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]
    for url in candidate_patterns:
        # Perform a HEAD request to see if the resource exists.
        try:
            response = requests.head(url, timeout=15)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def _download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1 << 14,
    headers: Mapping[str, str] | None = None,
) -> None:
    """
    Stream a remote resource to a destination path using optional request headers.

    Purpose:
        Provide a thin wrapper around `requests.get` that writes streamed content
        to disk while supporting caller-provided headers (e.g., CK-12 browser headers).

    Args:
        url (str): The fully qualified URL to download.
        dest (Path): Final destination file path for the downloaded content.
        chunk_size (int): Chunk size used when iterating over streamed content.
        headers (Mapping[str, str] | None): Optional headers forwarded to
            `requests.get`.

    Raises:
        requests.RequestException: If the HTTP request fails or response status is
            not OK.

    Side Effects:
        Creates parent directories when missing and writes a temporary file before
        atomically replacing the destination.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp_path = dest.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(dest)


def _copy_local_file(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def _extract_simple_wiki_bz2(bz2_path: Path) -> Path:
    """
    Extract a Simple Wiki `.bz2` archive to a `.xml` file via streaming copy.

    Purpose:
        Materialize a plain XML dump from the compressed `.bz2` archive while
        keeping memory usage bounded and writes atomic.

    Args:
        bz2_path (Path): Path to the `.bz2` archive to decompress.

    Returns:
        Path: The extracted `.xml` file path.

    Raises:
        OSError: If reading, writing, or replacing files fails.
        EOFError: If the `.bz2` stream is truncated.
        RuntimeError: If decompression fails due to invalid input.

    Side Effects:
        Writes a temporary XML file, then atomically replaces the final XML
        destination; cleans up the temp file on errors.
    """
    xml_path = bz2_path.with_suffix(".xml")
    tmp_path = xml_path.with_suffix(".xml.tmp")
    try:
        # Stream-decompress to a temp path to avoid loading the full dump in memory.
        with bz2.open(bz2_path, "rb") as source, tmp_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
        tmp_path.replace(xml_path)
    except Exception:
        # Clean up temp output so retries are safe and idempotent.
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return xml_path

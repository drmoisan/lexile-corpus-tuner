from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterator

import requests

RAW_ROOT = Path("data/corpus/raw")
GUTENBERG_DIR = RAW_ROOT / "gutenberg"
SIMPLE_WIKI_DIR = RAW_ROOT / "simple_wiki"
GUTENBERG_IDS_FILE = Path("data/meta/gutenberg_ids.txt")
OER_MANIFEST = Path("data/meta/oer_sources.json")
DEFAULT_SIMPLE_WIKI_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)


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
    """
    ensure_dirs()
    if dump_url is None:
        dump_url = os.environ.get(
            "LEXILE_SIMPLE_WIKI_DUMP_URL", DEFAULT_SIMPLE_WIKI_URL
        )
    filename = dump_url.split("/")[-1] or "simplewiki_dump.xml.bz2"
    dest = SIMPLE_WIKI_DIR / filename
    if dest.exists():
        LOGGER.info("Simple Wiki dump already exists at %s; skipping.", dest)
        return dest
    LOGGER.info("Downloading Simple Wiki dump from %s", dump_url)
    _download_file(dump_url, dest)
    return dest


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
        try:
            if url.startswith("file://"):
                _copy_local_file(Path(url[7:]), dest)
            elif url.startswith("/"):
                _copy_local_file(Path(url), dest)
            else:
                _download_file(url, dest)
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


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 14) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
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

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath


class EPUBParseError(RuntimeError):
    """Raised when an EPUB archive cannot be parsed."""


def extract_text_from_epub(epub_path: Path) -> str:
    """Return the concatenated text of all readable chapters in an EPUB."""
    if not epub_path.exists():
        raise EPUBParseError(f"EPUB file not found: {epub_path}")

    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            opf_path = _locate_opf(zf)
            spine_paths = _spine_items(zf, opf_path)
            if not spine_paths:
                spine_paths = _fallback_text_items(zf)
            texts: list[str] = []
            for rel_path in spine_paths:
                try:
                    raw_html = zf.read(rel_path).decode("utf-8", errors="ignore")
                except KeyError:
                    continue
                text = _html_to_text(raw_html)
                if text:
                    texts.append(text)
            return "\n\n".join(texts).strip()
    except zipfile.BadZipFile as exc:
        raise EPUBParseError(f"Invalid EPUB archive: {epub_path}") from exc


def _locate_opf(zf: zipfile.ZipFile) -> str:
    try:
        container_xml = zf.read("META-INF/container.xml")
    except KeyError as exc:
        raise EPUBParseError("EPUB missing META-INF/container.xml") from exc
    try:
        root = ET.fromstring(container_xml)
    except ET.ParseError as exc:
        raise EPUBParseError("Unable to parse container.xml") from exc
    rootfile = root.find(".//{*}rootfile")
    if rootfile is None:
        raise EPUBParseError("container.xml missing rootfile element")
    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        raise EPUBParseError("rootfile missing full-path attribute")
    return opf_path


def _spine_items(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    try:
        opf_xml = zf.read(opf_path)
    except KeyError:
        return []
    try:
        root = ET.fromstring(opf_xml)
    except ET.ParseError:
        return []
    manifest: dict[str, dict[str, str]] = {}
    manifest_el = root.find(".//{*}manifest")
    if manifest_el is not None:
        for item in manifest_el.findall("{*}item"):
            item_id = item.attrib.get("id")
            href = item.attrib.get("href")
            media_type = item.attrib.get("media-type", "")
            if item_id and href:
                manifest[item_id] = {"href": href, "media_type": media_type}

    spine_paths: list[str] = []
    spine_el = root.find(".//{*}spine")
    if spine_el is not None:
        for itemref in spine_el.findall("{*}itemref"):
            item_id = itemref.attrib.get("idref")
            if not item_id:
                continue
            manifest_item = manifest.get(item_id)
            if not manifest_item:
                continue
            media_type = manifest_item["media_type"].lower()
            if not _is_text_media(media_type):
                continue
            href = manifest_item["href"]
            resolved = _resolve_href(opf_path, href)
            spine_paths.append(resolved)
    return spine_paths


def _fallback_text_items(zf: zipfile.ZipFile) -> list[str]:
    text_suffixes = {".xhtml", ".html", ".htm", ".txt"}
    return [
        name
        for name in zf.namelist()
        if PurePosixPath(name).suffix.lower() in text_suffixes
    ]


def _resolve_href(opf_path: str, href: str) -> str:
    base = PurePosixPath(opf_path).parent
    if str(base) in ("", "."):
        return PurePosixPath(href).as_posix()
    return (base / PurePosixPath(href)).as_posix()


def _is_text_media(media_type: str) -> bool:
    return any(
        media_type.startswith(prefix)
        for prefix in ("application/xhtml", "text/html", "text/plain")
    )


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._last_was_newline = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._append_newline()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self._append_newline()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            if self._chunks and not self._chunks[-1].endswith((" ", "\n")):
                self._chunks.append(" ")
            self._chunks.append(text)
            self._last_was_newline = False

    def _append_newline(self) -> None:
        if not self._chunks or self._last_was_newline:
            return
        self._chunks.append("\n")
        self._last_was_newline = True

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [line.strip() for line in raw.splitlines()]
        filtered = "\n".join(line for line in lines if line)
        return filtered.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()

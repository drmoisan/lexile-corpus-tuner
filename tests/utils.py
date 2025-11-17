from __future__ import annotations

import zipfile
from pathlib import Path


def write_minimal_epub(
    path: Path, chapters: list[str], include_spine: bool = True
) -> None:
    """Create a minimal EPUB file with the provided XHTML chapters."""
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    manifest_items = []
    spine_items = []
    chapter_files = []
    for idx, chapter in enumerate(chapters, start=1):
        href = f"chapter{idx}.xhtml"
        manifest_items.append(
            f'<item id="chap{idx}" href="{href}" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="chap{idx}"/>')
        chapter_files.append((f"OEBPS/{href}", chapter))
    spine_block = (
        "<spine>" + "".join(spine_items) + "</spine>" if include_spine else "<spine/>"
    )
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
  </metadata>
  <manifest>
    {''.join(manifest_items)}
  </manifest>
  {spine_block}
</package>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf)
        for file_path, body in chapter_files:
            zf.writestr(file_path, body)

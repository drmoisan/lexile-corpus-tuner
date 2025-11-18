from __future__ import annotations

import argparse
import bz2
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO, Iterator

NAMESPACE = "{http://www.mediawiki.org/xml/export-0.10/}"


def open_dump(path: Path) -> IO[bytes]:
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def iter_articles(stream: IO[bytes]) -> Iterator[dict[str, str | int]]:
    context = ET.iterparse(stream, events=("end",))
    for _, elem in context:
        if elem.tag != f"{NAMESPACE}page":
            continue

        ns = elem.findtext(f"{NAMESPACE}ns")
        if ns != "0":
            elem.clear()
            continue
        if elem.find(f"{NAMESPACE}redirect") is not None:
            elem.clear()
            continue

        title = elem.findtext(f"{NAMESPACE}title") or ""
        page_id_text = elem.findtext(f"{NAMESPACE}id") or "0"
        try:
            page_id = int(page_id_text)
        except ValueError:
            page_id = 0

        revision = elem.find(f"{NAMESPACE}revision")
        text_elem = revision.find(f"{NAMESPACE}text") if revision is not None else None
        text = text_elem.text or ""

        yield {
            "id": page_id,
            "title": title,
            "text": text,
            "source_id": "simple_wiki",
        }
        elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Simple English Wikipedia articles into JSONL."
    )
    parser.add_argument("--dump", type=Path, required=True, help="Path to XML dump.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/raw/simple_wiki/simplewiki_articles.jsonl"),
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional limit for debugging/testing.",
    )
    args = parser.parse_args()

    if not args.dump.exists():
        print(f"Dump file not found: {args.dump}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        open_dump(args.dump) as stream,
        args.output.open("w", encoding="utf-8") as out_file,
    ):
        for article in iter_articles(stream):
            out_file.write(json.dumps(article, ensure_ascii=False))
            out_file.write("\n")
            written += 1
            if args.max_articles and written >= args.max_articles:
                break

    print(f"Wrote {written} Simple Wiki articles to {args.output}")


if __name__ == "__main__":
    main()

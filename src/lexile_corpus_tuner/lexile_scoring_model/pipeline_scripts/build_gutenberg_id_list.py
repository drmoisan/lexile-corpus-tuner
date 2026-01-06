from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
import requests

if TYPE_CHECKING:
    from collections.abc import Iterable

API_URL = "https://gutendex.com/books"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0


class ParquetStore(Protocol):
    """Persistence abstraction for storing incremental parquet output."""

    def exists(self) -> bool:
        """Return True if persisted parquet data is available."""
        ...  # pragma: no cover - protocol contract only

    def load(self) -> pd.DataFrame:
        """Load the persisted parquet data into a DataFrame."""
        ...  # pragma: no cover - protocol contract only

    def save(self, df: pd.DataFrame) -> None:
        """Persist the provided DataFrame as parquet."""
        ...  # pragma: no cover - protocol contract only


class FileParquetStore(ParquetStore):
    """Filesystem-backed parquet store used by the production CLI."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> pd.DataFrame:
        # pandas stub exposes read_parquet kwargs as Unknown values.
        return pd.read_parquet(self._path)  # type: ignore[reportUnknownMemberType]

    def save(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # pandas exposes to_parquet kwargs as Unknown; ignore for type checker only.
        df.to_parquet(self._path, index=False)  # type: ignore[reportUnknownMemberType]


def _resolve_parquet_store(
    parquet_path: Path | None, parquet_store: ParquetStore | None
) -> ParquetStore:
    """Return a parquet store, ensuring defaults map to filesystem storage."""

    if parquet_store is not None:
        return parquet_store
    if parquet_path is None:
        raise ValueError("parquet_path is required when parquet_store is not provided")
    return FileParquetStore(parquet_path)


def load_checkpoint(checkpoint_path: Path) -> int:
    """Load the last successful page from checkpoint."""
    if not checkpoint_path.exists():
        return 0

    try:
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data.get("last_page", 0)
    except Exception:
        return 0


def save_checkpoint(checkpoint_path: Path, page: int) -> None:
    """Save the current progress to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as fh:
        json.dump({"last_page": page}, fh)


def fetch_books_incremental(
    languages: Iterable[str],
    english_only: bool,
    checkpoint_path: Path,
    parquet_path: Path | None,
    *,
    parquet_store: ParquetStore | None = None,
) -> pd.DataFrame:
    """Fetch Gutenberg book metadata incrementally with checkpoint support.

    Args:
        languages: Language codes to filter by (e.g., ["en"])
        english_only: If True, exclude books with multiple languages
        checkpoint_path: Path to checkpoint file for resumable downloads
        parquet_path: Destination of the parquet file when using a filesystem store
        parquet_store: Optional custom parquet store for dependency injection

    Returns:
        DataFrame with book metadata
    """
    normalized_langs = sorted({lang.strip().lower() for lang in languages if lang})
    if not normalized_langs:
        raise ValueError("At least one language must be provided.")

    # Load previous progress
    last_page = load_checkpoint(checkpoint_path)

    store = _resolve_parquet_store(parquet_path, parquet_store)

    # Load existing data if available
    if store.exists():
        df = store.load()
        existing_count = len(df)
    else:
        df = pd.DataFrame()
        existing_count = 0

    print(
        f"Resuming from page {last_page + 1}, {existing_count} books already collected",
        file=sys.stderr,
    )

    params = {"languages": ",".join(normalized_langs)}
    url: str | None = API_URL
    page_count = 0

    # Skip to the last successful page
    while url and page_count < last_page:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        url = payload.get("next")
        params = None
        page_count += 1

    # Process remaining pages
    while url:
        retry_count = 0
        retry_delay = INITIAL_RETRY_DELAY
        page_books: list[dict[str, Any]] = []

        while retry_count < MAX_RETRIES:
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()

                # Process the page immediately
                payload = response.json()
                for entry in payload.get("results", []):
                    entry_langs = [code.lower() for code in entry.get("languages", [])]
                    if english_only and set(entry_langs) != {"en"}:
                        continue
                    if any(lang in normalized_langs for lang in entry_langs):
                        # Extract key fields from entry
                        book_data = {
                            "id": entry.get("id"),
                            "title": entry.get("title"),
                            "authors": ", ".join(
                                [
                                    author.get("name", "")
                                    for author in entry.get("authors", [])
                                ]
                            ),
                            "subjects": "; ".join(entry.get("subjects", [])),
                            "bookshelves": "; ".join(entry.get("bookshelves", [])),
                            "languages": ", ".join(entry.get("languages", [])),
                            "download_count": entry.get("download_count"),
                            "media_type": entry.get("media_type"),
                            "copyright": entry.get("copyright"),
                        }
                        page_books.append(book_data)

                # Success - append to DataFrame and save incrementally
                if page_books:
                    page_df = pd.DataFrame(page_books)
                    df = pd.concat([df, page_df], ignore_index=True)
                    store.save(df)

                page_count += 1
                save_checkpoint(checkpoint_path, page_count)

                if page_count % 10 == 0:
                    print(
                        f"Processed {page_count} pages, {len(df)} books total",
                        file=sys.stderr,
                    )

                url = payload.get("next")
                params = None
                break

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        print(
                            f"Rate limit exceeded after {MAX_RETRIES} retries "
                            f"at page {page_count + 1}.",
                            file=sys.stderr,
                        )
                        print(
                            f"Progress saved: {len(df)} books from "
                            f"{page_count} pages.",
                            file=sys.stderr,
                        )
                        print(
                            f"Run again to resume from page {page_count + 1}.",
                            file=sys.stderr,
                        )
                        return df
                    print(
                        f"Rate limited at page {page_count + 1}. "
                        f"Waiting {retry_delay:.1f}s before retry "
                        f"{retry_count}/{MAX_RETRIES}...",
                        file=sys.stderr,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

    print(f"Completed all pages. Total: {len(df)} books", file=sys.stderr)
    return df


def main() -> None:  # pragma: no cover - CLI orchestration
    parser = argparse.ArgumentParser(
        description="Fetch Gutenberg book metadata and generate ID list."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/gutenberg/gutenberg_ids.txt"),
        help="Where to write the ID list.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/meta/gutenberg/gutenberg_books.parquet"),
        help="Where to write the full book metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/meta/gutenberg/.gutenberg_checkpoint.json"),
        help="Checkpoint file for resumable downloads.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Gutendex language codes to include (default: en).",
    )
    parser.add_argument(
        "--allow-multi-language",
        action="store_true",
        help="Include works that list additional languages besides English.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = fetch_books_incremental(
            args.languages,
            english_only=not args.allow_multi_language,
            checkpoint_path=args.checkpoint,
            parquet_path=args.parquet,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Failed to fetch Gutenberg books: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write final outputs
    header_parts = [
        "Autogenerated from Gutendex",
        f"Languages: {', '.join(args.languages)}",
    ]
    if not args.allow_multi_language:
        header_parts.append("Filtered to strictly English entries")
    header = "# " + " | ".join(header_parts) + "\n"

    sorted_ids: list[int] = sorted(df["id"].tolist())  # type: ignore[arg-type]
    with args.output.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for ebook_id in sorted_ids:
            fh.write(f"{ebook_id}\n")

    print(f"Wrote {len(sorted_ids)} Gutenberg IDs to {args.output}")
    print(f"Full metadata saved to {args.parquet}")

    # Clean up checkpoint on successful completion
    if args.checkpoint.exists():
        args.checkpoint.unlink()
        print(f"Removed checkpoint file {args.checkpoint}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from scripts.dev_tools.json_config import iter_governed_files


class FormatResult:
    def __init__(self, changed: bool, failed: bool, messages: Iterable[str]):
        self.changed = changed
        self.failed = failed
        self.messages = list(messages)


def run_jq_format(path: Path, check: bool, jq_path: str) -> tuple[bool, bool, str]:
    """Format a single JSON file with jq --sort-keys.

    Returns (changed, failed, message).
    """
    proc = subprocess.run(  # noqa: S603 - fixed jq binary path, user input limited to known files
        [jq_path, "--sort-keys", ".", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, True, f"jq failed for {path}: {proc.stderr.strip()}"

    original = path.read_text()
    formatted = proc.stdout
    if original == formatted:
        return False, False, f"{path}: already formatted"
    if check:
        return True, False, f"{path}: would reformat"
    path.write_text(formatted)
    return True, False, f"{path}: reformatted"


def format_files(targets: Iterable[Path], check: bool, verbose: bool) -> FormatResult:
    changed = False
    failed = False
    messages: list[str] = []
    jq_path = shutil.which("jq")
    if jq_path is None:
        return FormatResult(False, True, ["jq executable not found on PATH"])

    for file_path in targets:
        if not file_path.is_file():
            continue
        c, f, msg = run_jq_format(file_path, check, jq_path)
        changed = changed or c
        failed = failed or f
        if verbose or f or c or check:
            messages.append(msg)
    return FormatResult(changed, failed, messages)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format governed JSON files with jq --sort-keys"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional specific files/dirs; defaults to governed globs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check; exit non-zero if changes needed",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file status")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path(__file__).resolve().parents[2]

    # If explicit paths provided, override governed files
    if args.paths:

        def iter_paths() -> Iterable[Path]:
            for p in args.paths:
                path = Path(p)
                if path.is_dir():
                    yield from path.rglob("*.json")
                else:
                    yield path

        target_files = list(iter_paths())
    else:
        target_files = list(iter_governed_files(root))

    result = format_files(target_files, check=args.check, verbose=args.verbose)
    for msg in result.messages:
        print(msg)
    if result.failed:
        return 1
    if args.check and result.changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

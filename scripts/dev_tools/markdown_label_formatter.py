"""Format markdown chat transcripts with labeled sections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

LABEL_PREFIXES: tuple[str, ...] = ("User:", "GitHub Copilot:")
SEPARATOR_LINE = "---"


def is_label_line(line: str) -> bool:
    """Return True when the line starts with a known label."""
    return line.startswith(LABEL_PREFIXES)


def is_separator_line(line: str) -> bool:
    """Return True when the line is blank or exactly the separator token."""
    stripped = line.strip()
    return stripped == "" or stripped == SEPARATOR_LINE


def format_label_heading(line: str) -> tuple[str, str]:
    """Convert a label line to an H1 heading and return remaining text.

    Args:
        line: A line starting with one of the supported labels.

    Returns:
        A tuple of the formatted heading line and any trailing text following
        the label.
    """

    label, _, trailing = line.partition(":")
    heading = f"# {label.strip()}:"
    return heading, trailing.lstrip()


def ensure_separator_block(output_lines: list[str]) -> None:
    """Ensure a blank/---/blank separator exists before the next label."""
    while output_lines and output_lines[-1].strip() == "":
        output_lines.pop()
    output_lines.extend(["", SEPARATOR_LINE, ""])


def prefix_content_line(line: str) -> str:
    """Prefix a non-label, non-separator line with a markdown quote marker."""
    return f"> {line}" if line else ">"


def process_markdown(content: str) -> str:
    """Process markdown text according to the requested formatting rules."""
    lines = content.splitlines()
    trailing_newline = content.endswith("\n")

    output_lines: list[str] = []

    for line in lines:
        if is_label_line(line):
            if output_lines:
                ensure_separator_block(output_lines)

            heading, trailing_text = format_label_heading(line)
            output_lines.append(heading)

            if trailing_text:
                output_lines.extend(["", ""])
                output_lines.append(prefix_content_line(trailing_text))
            continue

        if is_separator_line(line):
            output_lines.append(SEPARATOR_LINE if line.strip() else "")
            continue

        output_lines.append(prefix_content_line(line))

    result = "\n".join(output_lines)
    if trailing_newline:
        result += "\n"
    return result


def read_content(source: Path | None) -> str:
    """Read content from a file path or stdin when path is None."""
    if source is None:
        return sys.stdin.read()
    return source.read_text(encoding="utf-8")


def write_output(content: str, target: Path | None) -> None:
    """Write formatted content to a file path or stdout when path is None."""
    if target is None:
        sys.stdout.write(content)
        return
    target.write_text(content, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the formatter script."""
    parser = argparse.ArgumentParser(
        description=(
            "Format markdown chat transcripts by normalizing labels, separators, "
            "and quoted content."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Optional path to a markdown file. Reads stdin when omitted.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output path. Writes to stdout when omitted.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the formatter script."""
    args = parse_args(argv)
    try:
        input_text = read_content(args.path)
        formatted = process_markdown(input_text)
        write_output(formatted, args.output)
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

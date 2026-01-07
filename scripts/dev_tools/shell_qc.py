"""Run shell formatting, linting, and tests for repository shell scripts."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

EXCLUDED_DIRS = {".venv", ".git", "node_modules", "dist", "build"}
SEARCH_DIRS = ("tools", "scripts")
TEST_DIR_CANDIDATES = (Path("tests") / "shell", Path("tests") / "bash")


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def _extract_shebang_command(line: str) -> str | None:
    if not line.startswith("#!"):
        return None

    payload = line[2:].strip()
    if not payload:
        return None

    try:
        parts = shlex.split(payload)
    except ValueError:
        return None

    if not parts:
        return None

    command = Path(parts[0]).name
    if command == "env":
        parts = parts[1:]
        while parts and parts[0].startswith("-"):
            parts = parts[1:]
        if not parts:
            return None
        command = Path(parts[0]).name

    return command


def _has_shell_shebang(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().lower()
    except OSError:
        return False

    command = _extract_shebang_command(first_line)
    return command in {"bash", "sh"}


def _is_shell_script(path: Path) -> bool:
    if path.suffix.lower() == ".sh":
        return True

    return _has_shell_shebang(path)


def discover_shell_scripts(root: Path | None = None) -> list[Path]:
    """Discover shell scripts in tools/ and scripts/ under the given root."""

    base = root or Path.cwd()
    discovered: set[Path] = set()
    for folder in SEARCH_DIRS:
        search_root = base / folder
        if not search_root.exists():
            continue
        for path in _iter_files(search_root):
            if _is_shell_script(path):
                discovered.add(path)
    return sorted(discovered, key=lambda item: str(item))


def find_bats_test_dirs(root: Path | None = None) -> list[Path]:
    """Return existing bats test directories under tests/shell or tests/bash."""

    base = root or Path.cwd()
    matches: list[Path] = []
    for candidate in TEST_DIR_CANDIDATES:
        path = base / candidate
        if path.exists():
            matches.append(path)
    return matches


def _print_missing_tool(tool: str, package: str | None = None) -> None:
    package_name = package or tool
    print(f"Missing required tool: {tool}")
    print(
        "Devcontainer install (apt-get): "
        f"apt-get update && apt-get install -y {package_name}"
    )
    print(f"macOS (Homebrew): brew install {package_name}")
    print(
        f"Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y {package_name}"
    )
    print("On Windows, use WSL for best results.")


def _require_tool(tool: str, package: str | None = None) -> bool:
    if shutil.which(tool) is None:
        _print_missing_tool(tool, package)
        return False
    return True


def _run_command(command: Sequence[str]) -> int:
    completed = subprocess.run(list(command), check=False)  # noqa: S603
    return completed.returncode


def run_check(root: Path | None = None) -> int:
    """Run shfmt in diff mode and shellcheck across discovered scripts."""

    files = discover_shell_scripts(root)
    if not files:
        print("No shell scripts found; skipping.")
        return 0

    if not _require_tool("shfmt"):
        return 127
    if not _require_tool("shellcheck"):
        return 127

    exit_code = _run_command(["shfmt", "-d", *[str(path) for path in files]])

    shellcheck_exit = 0
    for path in files:
        shellcheck_exit = max(shellcheck_exit, _run_command(["shellcheck", str(path)]))

    return max(exit_code, shellcheck_exit)


def run_format(root: Path | None = None) -> int:
    """Run shfmt in write mode across discovered scripts."""

    files = discover_shell_scripts(root)
    if not files:
        print("No shell scripts found; skipping.")
        return 0

    if not _require_tool("shfmt"):
        return 127

    return _run_command(["shfmt", "-w", *[str(path) for path in files]])


def run_test(root: Path | None = None) -> int:
    """Run bats against tests/shell or tests/bash when available."""

    test_dirs = find_bats_test_dirs(root)
    if not test_dirs:
        print("No shell test directories found; skipping.")
        return 0

    if shutil.which("bats") is None:
        print("bats not installed; skipping shell tests.")
        return 0

    exit_code = 0
    for test_dir in test_dirs:
        exit_code = max(exit_code, _run_command(["bats", str(test_dir)]))
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the shell QC tool."""

    parser = argparse.ArgumentParser(
        description="Run shell script formatting, linting, and tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Run shfmt -d and shellcheck.")
    subparsers.add_parser("format", help="Format shell scripts with shfmt.")
    subparsers.add_parser("test", help="Run bats tests when available.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    if args.command == "check":
        return run_check()
    if args.command == "format":
        return run_format()
    if args.command == "test":
        return run_test()

    raise ValueError(f"Unsupported command: {args.command}")


def main_check() -> int:
    """Entry point for shell-qc-check."""

    return run_check()


def main_format() -> int:
    """Entry point for shell-qc-format."""

    return run_format()


def main_test() -> int:
    """Entry point for shell-qc-test."""

    return run_test()


if __name__ == "__main__":
    sys.exit(main())

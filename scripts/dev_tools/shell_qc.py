"""Run shell formatting, linting, and tests for repository shell scripts."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

EXCLUDED_DIRS = {".venv", ".git", "node_modules", "dist", "build"}
SEARCH_DIRS = ("tools", "scripts")
# Directories containing bash scripts to track for subprocess coverage.
# kcov needs --bash-parse-files-in-dir for scripts invoked via `bash script.sh`.
BASH_SCRIPT_DIRS = ("scripts/bash",)
TEST_DIR_CANDIDATES = (Path("tests") / "shell", Path("tests") / "bash")

# kcov writes a fixed Cobertura filename (cov.xml) when using --cobertura-only.
# Keep the output under artifacts/ so it stays gitignored and plays nicely with
# VS Code coverage extensions.
DEFAULT_KCOV_OUT_DIR = Path("artifacts") / "pester" / "kcov"

ToolName = Literal["bats", "kcov", "shellcheck", "shfmt"]


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield all files under root, skipping excluded directories."""

    # Walk the tree and filter excluded directories in-place to prune traversal.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def _extract_shebang_command(line: str) -> str | None:
    """Extract the interpreter command from a shebang line (e.g. bash/sh)."""

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
    """Return True when the file has a bash/sh shebang."""

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().lower()
    except OSError:
        return False

    command = _extract_shebang_command(first_line)
    return command in {"bash", "sh"}


def _is_shell_script(path: Path) -> bool:
    """Return True when a path looks like a shell script by suffix or shebang."""

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
    """Print installation hints for a missing CLI dependency."""

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


def _run_tool(tool: ToolName, args: Sequence[str]) -> int:
    """Run a validated external tool and return its exit code.

    Purpose:
        Centralize the repo's external tool execution while ensuring the
        executable is resolved to a full path via `shutil.which()`.

    Args:
        tool (ToolName): CLI tool to execute.
        args (Sequence[str]): Arguments passed to the tool.

    Returns:
        int: Exit code from the process.

    Side Effects:
        Executes a subprocess.
    """

    exe = shutil.which(tool)
    if exe is None:
        _print_missing_tool(tool)
        return 127

    completed = (
        subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [exe, *args],
            check=False,
        )
    )
    return int(completed.returncode)


def run_check(root: Path | None = None) -> int:
    """Run shfmt in diff mode and shellcheck across discovered scripts."""

    files = discover_shell_scripts(root)
    if not files:
        print("No shell scripts found; skipping.")
        return 0

    # Fail fast with a clear message if required tools are not available.
    if shutil.which("shfmt") is None:
        _print_missing_tool("shfmt")
        return 127
    if shutil.which("shellcheck") is None:
        _print_missing_tool("shellcheck")
        return 127

    # Convert paths once so the same argument list is used by all tool invocations.
    file_args = [str(path) for path in files]

    exit_code = _run_tool("shfmt", ["-d", *file_args])

    shellcheck_exit = 0
    # Lint each file independently to keep failures attributable.
    for path in files:
        shellcheck_exit = max(shellcheck_exit, _run_tool("shellcheck", [str(path)]))

    return max(exit_code, shellcheck_exit)


def run_format(root: Path | None = None) -> int:
    """Run shfmt in write mode across discovered scripts."""

    files = discover_shell_scripts(root)
    if not files:
        print("No shell scripts found; skipping.")
        return 0

    # Fail fast with a clear message if required tools are not available.
    if shutil.which("shfmt") is None:
        _print_missing_tool("shfmt")
        return 127

    # Convert paths once so the formatter sees stable arguments.
    file_args = [str(path) for path in files]

    return _run_tool("shfmt", ["-w", *file_args])


def run_test(root: Path | None = None) -> int:
    """Run bats against tests/shell or tests/bash when available."""

    return run_test_with_options(root=root)


def _prepare_kcov_output_dirs(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
    """Prepare the output directories used by kcov.

    Purpose:
        Ensure a clean, deterministic output location for kcov reports.

    Args:
        repo_root (Path): Repository root used for resolving relative paths.
        out_dir (Path): Directory (relative to repo_root or absolute) where the
            merged `cov.xml` output should be written.

    Returns:
        tuple[Path, Path]: (merged_output_dir, runs_dir)

    Side Effects:
        Deletes any previous coverage directory at `merged_output_dir`.
    """

    merged_output_dir = out_dir
    if not merged_output_dir.is_absolute():
        merged_output_dir = repo_root / merged_output_dir

    # Start from a clean slate so Coverage Gutters doesn't show stale results.
    shutil.rmtree(merged_output_dir, ignore_errors=True)
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = merged_output_dir / ".kcov_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return merged_output_dir, runs_dir


def _cleanup_kcov_runs_dir(runs_dir: Path) -> None:
    """Remove intermediate kcov run directories.

    Purpose:
        Avoid leaving multiple `cov.xml` files under the artifacts tree, which
        can confuse coverage viewers that discover reports by filename.

    Args:
        runs_dir (Path): The directory containing per-test-run kcov outputs.
    """

    shutil.rmtree(runs_dir, ignore_errors=True)


def _extract_cobertura_line_rate(cov_xml_path: Path) -> float | None:
    """Extract the line-rate attribute from a Cobertura XML coverage report.

    Purpose:
        Parse the top-level `line-rate` attribute from a kcov-generated cov.xml
        to enable a deterministic stdout summary.

    Args:
        cov_xml_path (Path): Path to the Cobertura XML file (cov.xml).

    Returns:
        float | None: The line-rate as a float (0.0-1.0), or None if the file
            does not exist or the attribute cannot be parsed.

    Notes:
        Uses regex to avoid XML parsing overhead; kcov output is well-formed
        and the line-rate attribute appears early in the file.
    """

    if not cov_xml_path.exists():
        return None

    try:
        content = cov_xml_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Match line-rate="0.75" or line-rate='0.75' at the coverage element level.
    match = re.search(r'line-rate=["\']([0-9.]+)["\']', content)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _print_coverage_summary(cov_xml_path: Path) -> None:
    """Print a concise Bash coverage summary to stdout.

    Purpose:
        Emit a single-line summary of Bash coverage after kcov completes,
        suitable for CI logs and developer feedback.

    Args:
        cov_xml_path (Path): Path to the Cobertura XML file (cov.xml).

    Side Effects:
        Prints to stdout. No-op if cov.xml is missing or unparseable.
    """

    line_rate = _extract_cobertura_line_rate(cov_xml_path)
    if line_rate is not None:
        percent = line_rate * 100
        print(f"Bash coverage (lines): {percent:.1f}%")


def run_test_with_options(
    root: Path | None = None,
    *,
    coverage: bool = False,
    kcov_out_dir: Path = DEFAULT_KCOV_OUT_DIR,
) -> int:
    """Run shell tests, optionally capturing coverage with kcov.

    Purpose:
        Provide a single entrypoint for `shell-qc test` while optionally
        enabling kcov-based coverage for Bash scripts.

    Args:
        root (Path | None): Repository root override (primarily for tests).
        coverage (bool): When True, run `bats` under `kcov --cobertura-only`
            and emit `cov.xml` for Coverage Gutters.
        kcov_out_dir (Path): Output directory for the merged kcov report.

    Returns:
        int: Process exit code (0 = success).

    Side Effects:
        When coverage is enabled, writes artifacts under `kcov_out_dir`.
    """

    repo_root = root or Path.cwd()
    test_dirs = find_bats_test_dirs(repo_root)
    if not test_dirs:
        print("No shell test directories found; skipping.")
        return 0

    if shutil.which("bats") is None:
        if coverage:
            print("bats not installed; cannot run shell tests with coverage.")
            return 127
        print("bats not installed; skipping shell tests.")
        return 0

    if not coverage:
        exit_code = 0
        # Run bats once per test directory to keep failures localized.
        for test_dir in test_dirs:
            exit_code = max(exit_code, _run_tool("bats", [str(test_dir)]))
        return exit_code

    if shutil.which("kcov") is None:
        print("kcov not installed; cannot run shell tests with coverage.")
        _print_missing_tool("kcov")
        return 127

    merged_output_dir, runs_dir = _prepare_kcov_output_dirs(repo_root, kcov_out_dir)

    # Run each bats directory under kcov and merge results into one report.
    run_dirs: list[Path] = []
    exit_code = 0
    # Run directories in order, stopping on first failure to keep logs readable.
    for test_dir in test_dirs:
        # Use a stable per-test-dir output folder so merges are predictable.
        run_dir = runs_dir / test_dir.name
        run_dirs.append(run_dir)

        # Capture Bash coverage. We scope coverage to repo scripts/tools and
        # exclude test sources themselves.
        include_patterns = ",".join(str(repo_root / folder) for folder in SEARCH_DIRS)
        # Tell kcov to parse bash scripts in the designated bash-only directories
        # so it can track coverage of scripts invoked as subprocesses.
        bash_parse_dirs = ",".join(
            str(repo_root / folder) for folder in BASH_SCRIPT_DIRS
        )
        cmd = [
            "--cobertura-only",
            f"--bash-parse-files-in-dir={bash_parse_dirs}",
            f"--include-pattern={include_patterns}",
            f"--exclude-pattern={repo_root / 'tests'}",
            str(run_dir),
            shutil.which("bats") or "bats",
            str(test_dir),
        ]

        exit_code = max(exit_code, _run_tool("kcov", cmd))
        if exit_code != 0:
            break

    if exit_code == 0 and run_dirs:
        # When there's only one run directory, kcov --merge with --cobertura-only
        # produces empty results (a kcov limitation). In this common case, we
        # simply copy the cov.xml to the expected location directly.
        if len(run_dirs) == 1:
            # Copy the single run's cov.xml to the expected merged location.
            src_cov = run_dirs[0] / "cov.xml"
            dst_dir = merged_output_dir / "kcov-merged"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src_cov.exists():
                shutil.copy2(src_cov, dst_dir / "cov.xml")
        else:
            # Merge multiple per-directory runs into one report directory.
            run_dir_args = [str(path) for path in run_dirs]
            merge_args = ["--merge", str(merged_output_dir), *run_dir_args]
            exit_code = _run_tool("kcov", merge_args)

    _cleanup_kcov_runs_dir(runs_dir)

    # Emit a concise coverage summary after all kcov operations complete.
    # kcov --merge creates a 'kcov-merged' subdirectory inside the output dir.
    if coverage and exit_code == 0:
        cov_xml_path = merged_output_dir / "kcov-merged" / "cov.xml"
        _print_coverage_summary(cov_xml_path)

    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the shell QC tool."""

    parser = argparse.ArgumentParser(
        description="Run shell script formatting, linting, and tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Run shfmt -d and shellcheck.")
    subparsers.add_parser("format", help="Format shell scripts with shfmt.")
    test_parser = subparsers.add_parser("test", help="Run bats tests when available.")
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Run bats under kcov and emit Cobertura (cov.xml) for Coverage Gutters. "
            f"Output directory defaults to: {DEFAULT_KCOV_OUT_DIR}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    if args.command == "check":
        return run_check()
    if args.command == "format":
        return run_format()
    if args.command == "test":
        return run_test_with_options(coverage=bool(getattr(args, "coverage", False)))

    raise ValueError(f"Unsupported command: {args.command}")


def main_check() -> int:
    """Entry point for shell-qc-check."""

    return run_check()


def main_format() -> int:
    """Entry point for shell-qc-format."""

    return run_format()


def main_test() -> int:
    """Entry point for shell-qc-test."""

    return run_test_with_options()


if __name__ == "__main__":
    sys.exit(main())

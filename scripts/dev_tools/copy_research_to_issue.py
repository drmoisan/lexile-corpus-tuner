"""Interactive helper: copy a chosen research doc into an active issue folder.

This script is designed for the repo's docs workflow:

1) The user selects a research document (any file).
2) The user selects an "active issue" document (typically a plan/spec file inside
   `docs/features/active/<issue>/...`).
3) The script copies the research document into the selected issue document's
   parent directory, naming it `research.md`.

The script prefers a GUI file picker (Tkinter) when available, but falls back to
prompting for paths in the terminal when Tkinter is not installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scripts.dev_tools.tk_dialog_helpers import (
    pick_file_with_tkinter,
    resolve_initial_dir,
    resolve_workspace_root,
)

RESEARCH_START_DIR = Path("artifacts") / "research"
ACTIVE_ISSUE_START_DIR = Path("docs") / "features" / "active"


class FileSystem(Protocol):
    """Minimal filesystem seam for deterministic unit tests."""

    def exists(self, path: Path) -> bool: ...

    def copy_file(self, src: Path, dest: Path) -> None: ...

    def resolve_path(self, path_str: str) -> Path: ...


@dataclass(frozen=True)
class RealFileSystem(FileSystem):
    """Real filesystem implementation for production use."""

    def exists(self, path: Path) -> bool:
        return path.exists()

    def copy_file(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    def resolve_path(self, path_str: str) -> Path:
        return Path(path_str)


def resolve_issue_parent_dir(issue_path: Path) -> Path:
    """Resolve the parent directory to copy `research.md` into.

    Purpose:
        Interpret a user-selected "active issue" path. If the selection is a
        file, the parent directory is the destination. If the selection is a
        directory, the directory itself is the destination.

    Args:
        issue_path (Path): Path selected by the user.

    Returns:
        Path: Destination directory.

    Raises:
        ValueError: If issue_path is empty.
    """

    if str(issue_path).strip() == "":
        raise ValueError("issue_path is empty")

    # Branch on the path type to match how users might select the issue context.
    if issue_path.is_dir():
        return issue_path
    return issue_path.parent


def build_destination_path(issue_parent_dir: Path) -> Path:
    """Build the destination path for the copied research document.

    Purpose:
        Centralize the output filename (`research.md`) so tests and future
        refactors only need to update it in one place.

    Args:
        issue_parent_dir (Path): Directory in which to place `research.md`.

    Returns:
        Path: Full destination path.
    """

    return issue_parent_dir / "research.md"


def copy_research_document(
    *,
    fs: FileSystem,
    research_path: Path,
    issue_path: Path,
    overwrite: bool,
) -> Path:
    """Copy the research document into the issue parent directory.

    Purpose:
        Perform the copy with explicit overwrite behavior and a testable
        filesystem seam.

    Args:
        fs (FileSystem): Filesystem adapter.
        research_path (Path): Source research document path.
        issue_path (Path): A file or folder inside the active issue area.
        overwrite (bool): Whether to overwrite an existing `research.md`.

    Returns:
        Path: The destination path (`.../research.md`).

    Raises:
        FileNotFoundError: If the research document does not exist.
        FileExistsError: If destination exists and overwrite is False.
    """

    if not fs.exists(research_path):
        raise FileNotFoundError(research_path)

    issue_parent_dir = resolve_issue_parent_dir(issue_path)
    dest_path = build_destination_path(issue_parent_dir)

    # Enforce overwrite policy explicitly for safety.
    if fs.exists(dest_path) and not overwrite:
        raise FileExistsError(dest_path)

    fs.copy_file(research_path, dest_path)
    return dest_path


def _prompt_yes_no(prompt: str) -> bool:
    """Prompt the user for a yes/no answer on stdin.

    Purpose:
        Provide a deterministic, cross-platform prompt when a GUI message box is
        unavailable.

    Args:
        prompt (str): Prompt text.

    Returns:
        bool: True if the user answered yes.
    """

    # Re-prompt until we can deterministically interpret the user's intent.
    while True:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _prompt_path_fallback(*, label: str, fs: FileSystem) -> Path:
    """Prompt the user for a path when GUI selection isn't available.

    Purpose:
        Keep the tool usable in headless environments (e.g., minimal Linux
        containers) where Tkinter may be unavailable.

    Args:
        label (str): Human-friendly name for the path (used in prompts).
        fs (FileSystem): Filesystem adapter used to build a Path.

    Returns:
        Path: The entered path.

    Raises:
        ValueError: If the user enters an empty path.
    """

    raw = input(f"Enter path to {label}: ").strip()
    if not raw:
        raise ValueError(f"No path provided for {label}")
    return fs.resolve_path(raw).expanduser()


def select_file(
    *,
    title: str,
    label: str,
    fs: FileSystem,
    initial_relative_dir: Path | None,
) -> Path:
    """Select a file path, preferring GUI but falling back to stdin prompts.

    Purpose:
        Provide a single selection API so the script's workflow doesn't care
        whether Tkinter is available.

    Args:
        title (str): Title for the GUI dialog.
        label (str): Label for the stdin prompt.
        fs (FileSystem): Filesystem adapter.

    Returns:
        Path: Selected file path.

    Raises:
        ValueError: If the selection is cancelled or empty.
    """

    workspace_root = resolve_workspace_root()
    initial_dir = (
        resolve_initial_dir(
            workspace_root=workspace_root,
            relative_start_dir=initial_relative_dir,
            exists=Path.exists,
        )
        if initial_relative_dir is not None
        else None
    )

    picked = pick_file_with_tkinter(title=title, initial_dir=initial_dir)
    if picked is not None:
        return picked
    return _prompt_path_fallback(label=label, fs=fs)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Purpose:
        Keep `main()` small and testable by isolating argument parsing.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="copy-research-to-issue",
        description=(
            "Interactively select a research document and an active issue file, "
            "then copy the research document into the issue folder as research.md."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing research.md without prompting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the interactive copy workflow.

    Purpose:
        Prompt the user to select (via GUI file picker when possible):
        1) A research document
        2) A file inside the target active issue folder
        Then copy the research document into that issue folder as `research.md`.

    Args:
        argv (list[str] | None): Optional argv override (defaults to sys.argv[1:]).

    Side Effects:
        Copies a file on disk (or via the provided filesystem adapter).
        Prints a short success message to stdout.
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv or [])

    fs = RealFileSystem()

    research_path = select_file(
        title="Select research document",
        label="research document",
        fs=fs,
        initial_relative_dir=RESEARCH_START_DIR,
    )

    issue_path = select_file(
        title="Select active issue file (e.g., issue.md)",
        label="active issue document",
        fs=fs,
        initial_relative_dir=ACTIVE_ISSUE_START_DIR,
    )

    overwrite = bool(args.overwrite)
    issue_parent_dir = resolve_issue_parent_dir(issue_path)
    dest_path = build_destination_path(issue_parent_dir)

    # If the destination exists and overwrite isn't forced, ask for confirmation.
    if fs.exists(dest_path) and not overwrite:
        overwrite = _prompt_yes_no(
            f"{dest_path} already exists. Overwrite?",
        )
        if not overwrite:
            print("Cancelled; existing research.md left unchanged.")
            return

    copied_to = copy_research_document(
        fs=fs,
        research_path=research_path,
        issue_path=issue_path,
        overwrite=overwrite,
    )
    print(f"Copied research to: {copied_to}")


if __name__ == "__main__":
    main(sys.argv[1:])

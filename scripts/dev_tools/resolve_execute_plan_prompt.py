"""Fill the execute-plan prompt with a resolved feature folder and copy it.

This helper resolves the active feature folder (or uses an explicit selection),
substitutes it into `.github/prompts/execute-plan-python-engineer.prompt.md`,
prints the result, and attempts to copy it to the clipboard for pasting into
Copilot Chat.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    """Return the UTF-8 contents of a file."""

    return path.read_text(encoding="utf-8")


def current_branch(workspace: Path) -> str | None:
    """Return the current git branch name, or None if unavailable."""

    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            return None
        result = subprocess.run(  # noqa: S603
            [git_executable, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    branch = result.stdout.strip()
    return branch or None


def normalize_branch_suffix(branch: str) -> str:
    """Strip common prefixes and issue markers from a branch name suffix."""

    suffix = branch.split("/")[-1]
    suffix = suffix.replace("#", "")
    suffix = re.sub(r"-?\d+$", "", suffix)
    return suffix


def list_feature_folders(active_dir: Path) -> list[str]:
    """Return sorted feature folder names under the active directory."""

    return sorted(entry.name for entry in active_dir.iterdir() if entry.is_dir())


def select_feature_folder(
    active_dir: Path, requested: str | None, branch: str | None
) -> str:
    """Choose a feature folder based on an explicit request or branch suffix."""

    candidates = list_feature_folders(active_dir)
    if not candidates:
        raise ValueError(f"No feature folders found under {active_dir}")

    if requested:
        if requested in candidates:
            return requested

        try:
            req_path = Path(requested).resolve()
            if req_path.is_relative_to(active_dir):
                rel = req_path.relative_to(active_dir)
                if rel.parts and rel.parts[0] in candidates:
                    return rel.parts[0]
        except (ValueError, OSError):
            pass

        raise ValueError(f"Feature folder '{requested}' not found under {active_dir}")

    if branch:
        suffix = normalize_branch_suffix(branch)
        matches: list[str] = []
        for name in candidates:
            if suffix and suffix in name:
                matches.append(name)
                continue
            if branch and branch in name:
                matches.append(name)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "Multiple feature folders match the current branch; "
                "rerun with --feature to disambiguate: " + ", ".join(matches)
            )

    raise ValueError(
        "Could not resolve feature folder automatically; "
        "provide one with --feature. Available: " + ", ".join(candidates)
    )


def replace_feature_token(prompt: str, feature_folder: str) -> str:
    """Replace <feature> tokens with the concrete folder name."""

    return prompt.replace("<feature>", feature_folder)


def replace_agent_token(prompt: str, agent: str) -> str:
    """Replace <agent> tokens with the selected agent name."""
    if not agent:
        return prompt
    return prompt.replace("<agent>", agent)


def copy_to_clipboard(text: str) -> bool:
    """Attempt to copy text to the clipboard using common tools.

    Returns True on success, False if no supported clipboard mechanism is found.
    """

    try:
        import pyperclip  # type: ignore
    except ImportError:
        pyperclip = None  # type: ignore

    pyperclip_error: Exception | None = None
    if pyperclip is not None:
        try:
            pyperclip.copy(text)
            return True
        except Exception as error:  # noqa: BLE001
            pyperclip_error = error

    commands: tuple[list[str], ...] = (
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["clip"],
    )

    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            subprocess.run(  # noqa: S603
                [executable, *command[1:]],
                input=text,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue

    if pyperclip_error is not None:
        print(f"pyperclip copy failed: {pyperclip_error}", file=sys.stderr)

    return False


def build_prompt_text(
    workspace: Path, feature_folder: str, prompt_path: Path, agent: str | None = None
) -> str:
    """Load the prompt file and substitute the feature folder."""

    prompt_text = read_text(prompt_path)
    prompt_text = replace_feature_token(prompt_text, feature_folder)
    if agent:
        prompt_text = replace_agent_token(prompt_text, agent)
    return prompt_text


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fill execute-plan prompt and copy it to the clipboard.",
    )
    parser.add_argument(
        "--feature",
        dest="feature",
        nargs="?",
        const="",
        default=None,
        help="Feature folder under docs/features/active (optional).",
    )
    parser.add_argument(
        "--agent",
        dest="agent",
        default=None,
        help="Agent name to inject into the template (optional).",
    )
    parser.add_argument(
        "--prompt-path",
        dest="prompt_path",
        default=".github/prompts/execute-plan-template.md",
        help="Path to the prompt template (relative to workspace).",
    )
    parser.add_argument(
        "--workspace",
        dest="workspace",
        default=None,
        help="Workspace root (defaults to repository root).",
    )
    parser.add_argument(
        "--no-copy",
        dest="no_copy",
        action="store_true",
        help="Print only; do not attempt clipboard copy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    requested_feature = args.feature or None
    workspace = (
        Path(args.workspace).resolve()
        if args.workspace
        else Path(__file__).resolve().parents[2]
    )
    active_dir = workspace / "docs" / "features" / "active"
    prompt_path = (workspace / args.prompt_path).resolve()

    if not prompt_path.is_file():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    branch = current_branch(workspace)
    try:
        feature_folder = select_feature_folder(active_dir, requested_feature, branch)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    prompt_text = build_prompt_text(
        workspace, feature_folder, prompt_path, agent=args.agent
    )
    print(prompt_text)

    if args.no_copy:
        return 0

    copied = copy_to_clipboard(prompt_text)
    if not copied:
        print(
            "Clipboard copy not available. Prompt printed for manual copy.",
            file=sys.stderr,
        )
        return 0

    print(
        f"Prompt copied to clipboard for feature folder '{feature_folder}'.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

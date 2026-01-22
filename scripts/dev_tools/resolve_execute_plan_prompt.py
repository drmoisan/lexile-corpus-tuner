"""Fill the execute-plan prompt with resolved variables and copy it to clipboard.

Purpose:
    This helper resolves the active feature folder from the target file path,
    substitutes variables into `.github/prompts/execute-plan-template.md`,
    prints the result, and attempts to copy it to the clipboard for pasting into
    Copilot Chat.

Supported variables:
    - ${file}: Workspace-relative path to the target plan file (forward slashes).
    - ${folderpath}: Workspace-relative folder path of the target file.
    - ${name}: Feature name derived from folder naming convention.
    - ${spec}: Path to spec.md under ${folderpath}.
    - ${research}: Path to research.md under ${folderpath}, annotated when missing.
    - ${user-story}: Path to user-story.md under ${folderpath}, section removed
      when missing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    """Return the UTF-8 contents of a file.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a string.
    """
    return path.read_text(encoding="utf-8")


def copy_to_clipboard(text: str) -> bool:
    """Attempt to copy text to the clipboard using common tools.

    Purpose:
        Provides cross-platform clipboard support using either pyperclip
        (if installed) or native system clipboard commands.

    Args:
        text: The text to copy to the clipboard.

    Returns:
        True on success, False if no supported clipboard mechanism is found.

    Side Effects:
        Writes to the system clipboard.
        Prints error to stderr if pyperclip fails.
    """
    try:
        import pyperclip  # type: ignore[import-untyped]
    except ImportError:
        pyperclip = None  # type: ignore[assignment]

    pyperclip_error: Exception | None = None
    if pyperclip is not None:
        try:
            pyperclip.copy(text)
            return True
        except Exception as error:  # noqa: BLE001 - CLI top-level error handling
            pyperclip_error = error

    # Fallback chain: try common clipboard commands across platforms
    commands: tuple[list[str], ...] = (
        ["pbcopy"],  # macOS
        ["wl-copy"],  # Wayland
        ["xclip", "-selection", "clipboard"],  # X11
        ["xsel", "--clipboard", "--input"],  # X11 alternative
        ["clip"],  # Windows
        ["clip.exe"],  # WSL
    )

    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            # S603: validated above via shutil.which
            subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
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


def strip_front_matter(content: str) -> str:
    """Remove YAML front matter from content if present.

    Purpose:
        Front matter is delimited by --- at start and end. This function
        removes it so the template content can be processed cleanly.

    Args:
        content: The raw content potentially containing front matter.

    Returns:
        Content with front matter removed.
    """
    lines = content.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1 :]).lstrip()
    return content


def _split_path_platform_agnostic(path_str: str) -> list[str]:
    """Split a path string into components, treating both '\\' and '/' as separators.

    Args:
        path_str: A path string with potentially mixed separators.

    Returns:
        A list of non-empty path components.
    """
    parts = [p for p in re.split(r"[\\/]+", path_str) if p]
    return parts


def _try_relative_to_workspace(path: Path, workspace_root: Path) -> Path:
    """Return path relative to workspace_root when possible.

    Args:
        path: The path to make relative.
        workspace_root: The workspace root directory.

    Returns:
        The relative path if possible, otherwise the original path.
    """
    try:
        return path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return path


def _resolve_folderpath(target_path: Path, workspace_root: Path) -> str:
    """Resolve ${folderpath} from a target file.

    Args:
        target_path: Path to the target file.
        workspace_root: The workspace root directory.

    Returns:
        The workspace-relative folder path as a string.
    """
    relative_target = _try_relative_to_workspace(target_path, workspace_root)
    folder = relative_target.parent
    return str(folder)


def _resolve_feature_foldername(folderpath: str) -> str:
    """Determine the feature folder name from folderpath.

    Purpose:
        Extract the feature folder name, handling versioned plan folders.

    Rules:
        - Split folderpath into components using platform-agnostic delimiters.
        - If the leaf folder starts with 'v', treat it as a versioned plan
          folder and use the parent folder as the feature folder.

    Args:
        folderpath: The folder path to analyze.

    Returns:
        The feature folder name.

    Raises:
        ValueError: If folderpath is empty.
    """
    parts = _split_path_platform_agnostic(folderpath)
    if not parts:
        raise ValueError("folderpath is empty")

    leaf = parts[-1]
    # If leaf starts with 'v' (e.g., v1, v2), it's a versioned plan folder
    if leaf.startswith("v") and len(parts) >= 2:
        return parts[-2]
    return leaf


def _resolve_name_from_feature_foldername(feature_foldername: str) -> str:
    """Extract ${name} from a feature folder name.

    Purpose:
        Feature folders follow the convention:
            yyyy-MM-dd-${name}-${issue}
        where ${name} may contain hyphens.

    Args:
        feature_foldername: The feature folder name to parse.

    Returns:
        The extracted feature name, or the original folder name if the
        convention doesn't match.
    """
    parts = feature_foldername.split("-")

    # Check if folder follows yyyy-MM-dd-name-issue convention
    if (
        len(parts) >= 5
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and len(parts[2]) == 2
    ):
        if (
            parts[0].isdigit()
            and parts[1].isdigit()
            and parts[2].isdigit()
            and parts[-1].isdigit()
        ):
            # Extract name parts between date and issue number
            name_parts = parts[3:-1]
            if name_parts:
                return "-".join(name_parts)

    return feature_foldername


def _resolve_spec_path(folderpath: str) -> str:
    """Resolve ${spec} as ${folderpath}/spec.md.

    Args:
        folderpath: The folder path containing the spec.

    Returns:
        The path to spec.md.
    """
    return str(Path(folderpath) / "spec.md")


def _resolve_research_value(folderpath: str, workspace_root: Path) -> str:
    """Resolve ${research} with existence awareness.

    Purpose:
        Returns the path to research.md, annotated if the file doesn't exist.

    Args:
        folderpath: The folder path containing research.md.
        workspace_root: The workspace root directory.

    Returns:
        The path to research.md, with "(missing)" suffix if it doesn't exist.
    """
    rel_research = Path(folderpath) / "research.md"
    full_research = workspace_root / rel_research

    if full_research.exists():
        return str(rel_research)

    return f"{rel_research} (missing)"


def _resolve_user_story_value(folderpath: str, workspace_root: Path) -> str:
    """Resolve ${user-story} with existence awareness.

    Purpose:
        Returns the path to user-story.md, annotated if the file doesn't exist.

    Args:
        folderpath: The folder path containing user-story.md.
        workspace_root: The workspace root directory.

    Returns:
        The path to user-story.md, with "(missing)" suffix if it doesn't exist.
    """
    rel_story = Path(folderpath) / "user-story.md"
    full_story = workspace_root / rel_story

    if full_story.exists():
        return str(rel_story)

    return f"{rel_story} (missing)"


def _remove_user_story_section_when_missing(template: str) -> str:
    """Remove the user-story authoritative document section from the template.

    Purpose:
        When user-story.md is absent, we remove authoritative document 4
        entirely from the prompt to avoid instructing agents to read a
        document that does not exist. Preserves a blank line before the
        next section.

    Args:
        template: The template content to modify.

    Returns:
        The template with the user-story section removed.
    """
    # Remove the entire section containing authoritative document 4 (user-story)
    # Pattern matches: 4. **User Story** ... `${user-story}` and any trailing blank line
    # Replace with a single newline to preserve spacing before next section
    pattern = r"^\s*4\.\s*\*\*User Story\*\*.*\n\s*`\$\{user-story\}`\s*\n?"
    return re.sub(pattern, "\n", template, flags=re.MULTILINE)


def _remove_user_story_clause_when_missing(template: str) -> str:
    """Remove the user-story clause from the template when no user story exists.

    Purpose:
        Removes references to "User Story" in prose when the file is missing.

    Args:
        template: The template content to modify.

    Returns:
        The template with user-story references cleaned up.
    """
    template = template.replace(" and User Story", "")
    template = template.replace(" and the User Story", "")
    template = template.replace("and User Story", "")
    return template


def _extract_template_variables(template: str) -> set[str]:
    """Extract variable names from ${var} placeholders in a template.

    Args:
        template: The template content to scan.

    Returns:
        A set of variable names found in the template.
    """
    return {m.group(1) for m in re.finditer(r"\$\{([^}]+)\}", template)}


def _replace_all_variables(template: str, variables: dict[str, str]) -> str:
    """Replace all ${var} placeholders in template using the provided mapping.

    Purpose:
        Performs variable substitution and validates that all placeholders
        are resolved.

    Args:
        template: The template content with placeholders.
        variables: A mapping of variable names to their values.

    Returns:
        The template with all placeholders replaced.

    Raises:
        ValueError: If any placeholder in the template cannot be resolved.
    """
    referenced = _extract_template_variables(template)
    missing = sorted(v for v in referenced if v not in variables)
    if missing:
        raise ValueError(f"Unresolved template variables: {', '.join(missing)}")

    resolved = template

    for key in sorted(referenced):
        resolved = resolved.replace(f"${{{key}}}", variables[key])

    if _extract_template_variables(resolved):
        raise ValueError("Template resolution failed: unresolved placeholders remain")

    return resolved


def build_prompt_text(
    workspace: Path, target_path: Path, prompt_path: Path, agent: str | None = None
) -> str:
    """Load the prompt file and substitute all variables.

    Purpose:
        Reads the prompt template, resolves all ${var} placeholders based on
        the target file path, and handles optional agent injection.

    Args:
        workspace: Workspace root path.
        target_path: Path to the target plan file.
        prompt_path: Path to the prompt template.
        agent: Optional agent name to inject.

    Returns:
        Resolved prompt text with all variables substituted.

    Raises:
        ValueError: If any template variable cannot be resolved.
    """
    content = read_text(prompt_path)
    content = strip_front_matter(content)

    # Resolve ${file} - workspace-relative path with forward slashes
    relative_target = _try_relative_to_workspace(target_path, workspace)
    file_str = str(relative_target).replace("\\", "/")

    # Resolve folder-based variables
    folderpath = _resolve_folderpath(target_path, workspace)
    feature_foldername = _resolve_feature_foldername(folderpath)
    name = _resolve_name_from_feature_foldername(feature_foldername)

    # Resolve user-story with existence check
    user_story_value = _resolve_user_story_value(folderpath, workspace)

    # If user story is missing, remove the entire section from the template
    if "(missing)" in user_story_value:
        content = _remove_user_story_section_when_missing(content)
        content = _remove_user_story_clause_when_missing(content)

    # Build variable mapping
    variables: dict[str, str] = {
        "file": file_str,
        "folderpath": folderpath,
        "name": name,
        "spec": _resolve_spec_path(folderpath),
        "research": _resolve_research_value(folderpath, workspace),
    }

    # Only include user-story in variables if not missing
    if "(missing)" not in user_story_value:
        variables["user-story"] = user_story_value

    resolved = _replace_all_variables(content, variables)

    # Replace agent token if provided
    if agent:
        resolved = resolved.replace("<agent_type>", agent)

    return resolved


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments (excluding program name).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fill execute-plan prompt and copy it to the clipboard.",
    )
    parser.add_argument(
        "--feature",
        dest="feature",
        default=None,
        help="Path to the target plan file (e.g., plan.md).",
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
    """CLI entry point.

    Purpose:
        Parses arguments, resolves the prompt template with variables,
        prints the result, and copies to clipboard.

    Args:
        argv: Command-line arguments (excluding program name).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    workspace = (
        Path(args.workspace).resolve()
        if args.workspace
        else Path(__file__).resolve().parents[2]
    )

    prompt_path = (workspace / args.prompt_path).resolve()
    if not prompt_path.is_file():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    if not args.feature:
        print("Error: --feature argument is required", file=sys.stderr)
        return 1

    target_path = Path(args.feature).resolve()
    if not target_path.exists():
        print(f"Target file not found: {target_path}", file=sys.stderr)
        return 1

    try:
        prompt_text = build_prompt_text(
            workspace, target_path, prompt_path, agent=args.agent
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

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

    print("Prompt copied to clipboard.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

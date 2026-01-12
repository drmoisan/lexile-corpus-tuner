"""scripts.dev_tools.resolve_file_prompt

Resolves ${...} variables in a prompt template and copies the result to clipboard.

Supported variables:
    - ${file}: Workspace-relative path to the target file (forward slashes).
    - ${folderpath}: Workspace-relative folder path of the target file.
    - ${name}: Feature name derived from folder naming convention.
    - ${spec}: Path to spec.md under ${folderpath}.
    - ${user-story}: Path to user-story.md under ${folderpath}, annotated when missing.

Usage:
    python resolve_file_prompt.py \\
        --template <path_to_template> \\
        --target <path_to_target_file>
"""

import argparse
import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path


def _missing_pyperclip_copy(text: str) -> None:
    """
    Raise a clear error when clipboard support is unavailable.

    Purpose:
        Provide a predictable failure mode when pyperclip is not installed so
        callers understand why clipboard operations fail.

    Args:
        text (str): The text that would have been copied to the clipboard.

    Returns:
        None: This function always raises and never returns.

    Raises:
        RuntimeError: Always raised to signal that pyperclip is missing.

    Side Effects:
        Raises an exception to stop the calling workflow when clipboard access
        is required.
    """
    raise RuntimeError(
        "Clipboard support requires the optional 'pyperclip' dependency."
    )


def copy_to_clipboard(text: str) -> bool:
    """Attempt to copy text to the clipboard.

    Purpose:
        VS Code tasks often run inside Linux containers where `pyperclip` may be
        installed but not usable (it depends on external clipboard tools). This
        function provides a robust best-effort copy that:
        1) Tries pyperclip first (when available).
        2) Falls back to common platform clipboard commands.

    Args:
        text (str): Text to copy.

    Returns:
        bool: True if text was copied to a clipboard mechanism, else False.

    Side Effects:
        May invoke a platform clipboard command via subprocess.
    """

    pyperclip_exception_type = getattr(pyperclip, "PyperclipException", None)
    exception_types: list[type[BaseException]] = [OSError, RuntimeError]
    if isinstance(pyperclip_exception_type, type) and issubclass(
        pyperclip_exception_type, Exception
    ):
        exception_types.append(pyperclip_exception_type)
    pyperclip_copy_exceptions = tuple(exception_types)

    try:
        pyperclip.copy(text)
        return True
    except pyperclip_copy_exceptions as error:
        pyperclip_error = error

    commands: tuple[list[str], ...] = (
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["clip"],
        ["clip.exe"],
    )

    # Try known clipboard commands. We resolve the executable path first to
    # avoid partial-path execution issues.
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
                [executable, *command[1:]],
                input=text,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue

    print(f"pyperclip copy failed: {pyperclip_error}", file=sys.stderr)

    return False


# Resolve pyperclip at runtime so the module can be imported without the
# optional dependency installed.
_pyperclip_spec = importlib.util.find_spec("pyperclip")
if _pyperclip_spec is None:
    pyperclip = types.SimpleNamespace(copy=_missing_pyperclip_copy)
else:
    pyperclip = importlib.import_module("pyperclip")


def strip_front_matter(content: str) -> str:
    """
    Removes YAML front matter from content if present.

    Front matter is delimited by --- at start and end.

    Args:
        content (str): The raw content potentially containing front matter.

    Returns:
        str: Content with front matter removed.
    """
    lines = content.split("\n")
    if lines and lines[0].strip() == "---":
        # Find the closing ---
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                # Return everything after the closing delimiter
                return "\n".join(lines[i + 1 :]).lstrip()
    return content


def _split_path_platform_agnostic(path_str: str) -> list[str]:
    """Split a path string into components, treating both '\\' and '/' as separators.

    Purpose:
        The resolver needs to operate consistently across Windows and Unix-like
        platforms. The folderpath it derives may be rendered with either
        separator depending on context.

    Args:
        path_str (str): A path string (typically workspace-relative).

    Returns:
        list[str]: Non-empty path components.
    """
    # Split on either separator to stay platform-agnostic.
    parts = [p for p in re.split(r"[\\/]+", path_str) if p]
    return parts


def _try_relative_to_workspace(path: Path, workspace_root: Path) -> Path:
    """Return path relative to workspace_root when possible.

    Purpose:
        Most prompts should refer to workspace-relative paths. If the target is
        outside the workspace, fall back to the original path.

    Args:
        path (Path): Path to relativize.
        workspace_root (Path): Workspace root (usually the repo root).

    Returns:
        Path: A relative path when possible, else the original path.
    """
    try:
        return path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return path


def _resolve_folderpath(target_path: Path, workspace_root: Path) -> str:
    """Resolve ${folderpath} from a target file.

    Args:
        target_path (Path): The target file path.
        workspace_root (Path): Workspace root used for relative resolution.

    Returns:
        str: Workspace-relative folder path of the target.
    """
    relative_target = _try_relative_to_workspace(target_path, workspace_root)
    folder = relative_target.parent
    return str(folder)


def _resolve_feature_foldername(folderpath: str) -> str:
    """Determine the feature folder name from folderpath.

    Rules:
        - Split folderpath into components using platform-agnostic delimiters.
        - If the leaf folder starts with 'v', treat it as a versioned plan
          folder and use the parent folder as the feature folder.

    Args:
        folderpath (str): Workspace-relative folder path.

    Returns:
        str: Feature folder name (the folder containing the feature docs).

    Raises:
        ValueError: If folderpath is empty or cannot be parsed.
    """
    parts = _split_path_platform_agnostic(folderpath)
    if not parts:
        raise ValueError("folderpath is empty")

    leaf = parts[-1]
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
        feature_foldername (str): Feature folder name.

    Returns:
        str: Extracted name portion, or the original folder name if it does not
        match the expected pattern.
    """
    parts = feature_foldername.split("-")

    # Decision logic:
    # - If the folder name matches the date prefix and has a trailing issue
    #   token, extract the middle as the name.
    # - Otherwise, fall back to the whole folder name.
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
            name_parts = parts[3:-1]
            if name_parts:
                return "-".join(name_parts)

    return feature_foldername


def _resolve_spec_path(folderpath: str) -> str:
    """Resolve ${spec} as ${folderpath}/spec.md using the OS delimiter."""
    return str(Path(folderpath) / "spec.md")


def _resolve_user_story_value(folderpath: str, workspace_root: Path) -> str:
    """Resolve ${user-story} with existence awareness.

    Args:
        folderpath (str): Workspace-relative folder path.
        workspace_root (Path): Workspace root used for existence checks.

    Returns:
        str: A user-story path string. If the file is missing, the string is
        annotated with a clear marker.
    """
    rel_story = Path(folderpath) / "user-story.md"
    full_story = workspace_root / rel_story

    if full_story.exists():
        return str(rel_story)

    return f"{rel_story} (missing)"


def _remove_user_story_clause_when_missing(template: str) -> str:
    """Remove the user-story clause from the template when no user story exists.

    Purpose:
        Some prompts include a natural-language clause that assumes a user story
        exists (e.g., "and the `${user-story}`"). When the file is absent, we
        delete the clause entirely to avoid confusing instructions.

    Args:
        template (str): Prompt template content.

    Returns:
        str: Updated template content with the clause removed.
    """
    return template.replace(" and the `${user-story}`", "")


def _extract_template_variables(template: str) -> set[str]:
    """Extract variable names from ${var} placeholders in a template."""
    return {m.group(1) for m in re.finditer(r"\$\{([^}]+)\}", template)}


def _replace_all_variables(template: str, variables: dict[str, str]) -> str:
    """Replace all ${var} placeholders in template using the provided mapping.

    Raises:
        ValueError: If any placeholder in the template cannot be resolved.
    """
    referenced = _extract_template_variables(template)
    missing = sorted(v for v in referenced if v not in variables)
    if missing:
        raise ValueError(f"Unresolved template variables: {', '.join(missing)}")

    resolved = template

    # Apply substitutions deterministically (sorted key order for stability).
    for key in sorted(referenced):
        resolved = resolved.replace(f"${{{key}}}", variables[key])

    # Safety check: no placeholders remain.
    if _extract_template_variables(resolved):
        raise ValueError("Template resolution failed: unresolved placeholders remain")

    return resolved


def resolve_prompt(template_content: str, target_path: Path, cwd: Path) -> str:
    """
    Substitutes ${file} in the template content.

    Uses the workspace-relative path of the target.

    Args:
        template_content (str): The raw content of the prompt template.
        target_path (Path): The path to the file to inject into the prompt.
        cwd (Path): The current working directory (used to calculate relative path).

    Returns:
        str: The resolved prompt content.
    """
    # Strip front matter first
    content = strip_front_matter(template_content)

    relative_target = _try_relative_to_workspace(target_path, cwd)

    # Keep ${file} forward-slashed to match existing prompt style.
    file_str = str(relative_target).replace("\\", "/")

    folderpath = _resolve_folderpath(target_path, cwd)
    feature_foldername = _resolve_feature_foldername(folderpath)
    name = _resolve_name_from_feature_foldername(feature_foldername)

    variables: dict[str, str] = {
        "file": file_str,
        "folderpath": folderpath,
        "name": name,
        "spec": _resolve_spec_path(folderpath),
        "user-story": _resolve_user_story_value(folderpath, cwd),
    }

    # If the user story is missing, remove the specific clause that references it.
    # This keeps the prompt deterministic and avoids instructing agents to read a
    # document that does not exist.
    if "(missing)" in variables["user-story"]:
        content = _remove_user_story_clause_when_missing(content)

    return _replace_all_variables(content, variables)


def main() -> None:
    """CLI entry point for the prompt resolver."""
    parser = argparse.ArgumentParser(description="Resolve ${file} in prompt template")
    parser.add_argument(
        "--template", required=True, help="Path to the prompt template file"
    )
    parser.add_argument(
        "--target", required=True, help="Path to the target file to be substituted"
    )

    args = parser.parse_args()

    template_path = Path(args.template)
    target_path = Path(args.target)

    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Read the template
        content = template_path.read_text(encoding="utf-8")

        # Resolve
        resolved_content = resolve_prompt(content, target_path, Path.cwd())

        copied = copy_to_clipboard(resolved_content)

        if copied:
            print("Successfully resolved prompt and copied to clipboard.")
            return

        # Fall back to printing the prompt so the task still succeeds even when
        # the container has no clipboard integration.
        print(
            "Clipboard copy not available; printing resolved prompt to stdout.",
            file=sys.stderr,
        )
        print(resolved_content)
        return

    except Exception as e:
        print(f"Error processing prompt: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

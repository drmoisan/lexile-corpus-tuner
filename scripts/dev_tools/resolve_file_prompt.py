"""
Resolves ${file} variable in a prompt template and copies the result to clipboard.

Usage:
    python resolve_file_prompt.py \\
        --template <path_to_template> \\
        --target <path_to_target_file>
"""

import argparse
import sys
from pathlib import Path

import pyperclip


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


def build_context_injection(target_path: Path) -> str:
    """
    Builds additional context if spec.md and user-story.md exist alongside target.

    Args:
        target_path (Path): The target file being resolved.

    Returns:
        str: Context injection text or empty string.
    """
    # Only inject context for plan.md files
    if target_path.name != "plan.md":
        return ""

    target_dir = target_path.parent
    spec_path = target_dir / "spec.md"
    user_story_path = target_dir / "user-story.md"

    if spec_path.exists() and user_story_path.exists():
        relative_spec = spec_path.relative_to(target_path.parent.parent.parent)
        relative_story = user_story_path.relative_to(target_path.parent.parent.parent)
        spec_str = str(relative_spec).replace("\\", "/")
        story_str = str(relative_story).replace("\\", "/")

        return (
            f"\n\n## Authoritative Requirements\n\n"
            f"This plan must fully deliver on the requirements defined in:\n"
            f"- `{spec_str}` - Technical specification and implementation details\n"
            f"- `{story_str}` - User stories and acceptance criteria\n\n"
            f"Read both documents thoroughly before generating the plan. "
            f"The plan must be sufficient to satisfy all requirements in these "
            f"authoritative sources."
        )
    return ""


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

    try:
        # adaptable: try to resolve relative to cwd
        # Note: We resolve both to ensure we are comparing absolute paths
        relative_target = target_path.resolve().relative_to(cwd.resolve())
    except ValueError:
        # fallback if file is outside cwd
        relative_target = target_path

    # Perform substitution; force forward slashes for prompt consistency
    path_str = str(relative_target).replace("\\", "/")
    resolved = content.replace("${file}", path_str)

    # Inject context about spec.md and user-story.md if applicable
    context_injection = build_context_injection(target_path)
    if context_injection:
        resolved += context_injection

    return resolved


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

        # Copy to clipboard
        pyperclip.copy(resolved_content)
        print("Successfully resolved prompt and copied to clipboard.")

    except Exception as e:
        print(f"Error processing prompt: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

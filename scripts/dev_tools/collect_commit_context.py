"""Collect Git commit context for commit message generation.

This script gathers staged changes and repository state information
to provide context for generating conventional commit messages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_git(args: list[str], allow_error: bool = False) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=not allow_error,
        )
        stdout = result.stdout or ""
        return stdout.strip()
    except subprocess.CalledProcessError as e:
        if allow_error:
            stdout = e.stdout or ""
            return stdout.strip()
        raise


def collect_commit_context(output_path: Path) -> None:
    """Collect commit context and write to output file."""
    sections: list[str] = []

    sections.append("Please generate a commit message based on the following content:")
    sections.append("\n")

    # Repository remotes
    sections.append("===== Repository remotes =====")
    sections.append("")
    remotes = run_git(["remote", "-v"])
    sections.append(remotes)
    sections.append("")

    # Current branch
    sections.append("===== Current branch =====")
    sections.append("")
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    sections.append(branch)
    sections.append("")

    # Upstream
    sections.append("===== Upstream =====")
    sections.append("")
    upstream = run_git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], allow_error=True
    )
    sections.append(upstream if upstream else "(no upstream)")
    sections.append("")

    # Status (short)
    sections.append("===== Status (short) =====")
    sections.append("")
    status = run_git(["status", "-sb"])
    sections.append(status)
    sections.append("")

    # Staged files (name-status)
    sections.append("===== Staged files (name-status) =====")
    sections.append("")
    staged = run_git(["diff", "--cached", "--name-status"], allow_error=True)
    sections.append(staged if staged else "(no staged changes)")
    sections.append("")

    # Staged diff
    sections.append("===== Staged diff =====")
    sections.append("")
    staged_diff = run_git(["diff", "--cached"], allow_error=True)
    sections.append(staged_diff if staged_diff else "(no staged changes)")
    sections.append("")

    # Unstaged files (name-status)
    sections.append("===== Unstaged files (name-status) =====")
    sections.append("")
    unstaged = run_git(["diff", "--name-status"], allow_error=True)
    sections.append(unstaged if unstaged else "(no unstaged changes)")
    sections.append("")

    # Unstaged diff
    sections.append("===== Unstaged diff =====")
    sections.append("")
    unstaged_diff = run_git(["diff"], allow_error=True)
    sections.append(unstaged_diff if unstaged_diff else "(no unstaged changes)")
    sections.append("")

    # Untracked files
    sections.append("===== Untracked files =====")
    sections.append("")
    untracked = run_git(
        ["ls-files", "--others", "--exclude-standard"], allow_error=True
    )
    sections.append(untracked if untracked else "(no untracked files)")
    sections.append("")

    # Diff stat (combined)
    sections.append("===== Diff stat (staged + unstaged) =====")
    sections.append("")
    diff_stat = run_git(["diff", "HEAD", "--stat"], allow_error=True)
    sections.append(diff_stat if diff_stat else "(no changes)")
    sections.append("")

    # Changed Python files
    sections.append("===== Changed Python files =====")
    sections.append("")
    all_changed = run_git(["diff", "HEAD", "--name-only"], allow_error=True)
    py_files = (
        [f for f in all_changed.split("\n") if f.endswith(".py")] if all_changed else []
    )
    sections.append("\n".join(py_files) if py_files else "(no Python files changed)")
    sections.append("")

    # Last commit (header only)
    sections.append("===== Last commit (header only) =====")
    sections.append("")
    last_commit = run_git(
        ["log", "-1", "--format=%H%n%aN <%aE>%n%aD%n%cN <%cE>%n%cD%n%s%n%b"],
        allow_error=True,
    )
    if last_commit:
        lines = last_commit.split("\n")
        sections.append(f"commit {lines[0]}")
        if len(lines) > 1:
            sections.append(f"Author:     {lines[1]}")
        if len(lines) > 2:
            sections.append(f"AuthorDate: {lines[2]}")
        if len(lines) > 3:
            sections.append(f"Commit:     {lines[3]}")
        if len(lines) > 4:
            sections.append(f"CommitDate: {lines[4]}")
        if len(lines) > 5:
            sections.append("")
            sections.append(f"    {lines[5]}")
            for line in lines[6:]:
                if line.strip():
                    sections.append(f"    {line}")
    else:
        sections.append("(no previous commits)")
    sections.append("")

    # Change intent (editable section)
    sections.append("===== Change intent (edit below) =====")
    sections.append("- What/why summary:")
    sections.append("- Breaking changes:")
    sections.append("- Affected modules:")
    sections.append("- Issue/PR refs:")
    sections.append("")

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Commit context written to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect Git commit context for commit message generation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/commit_context.txt"),
        help="Output file path (default: artifacts/commit_context.txt)",
    )
    args = parser.parse_args(argv)

    try:
        collect_commit_context(args.output)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

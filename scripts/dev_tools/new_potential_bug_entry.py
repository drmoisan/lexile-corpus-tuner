"""Create a potential bug entry and optionally open it in VS Code."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

SHORT_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _resolve_workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_short_name(short_name: str) -> None:
    if not short_name or not SHORT_NAME_PATTERN.fullmatch(short_name):
        raise ValueError(
            f"Aborted: '{short_name}' is invalid. Use kebab-case letters/numbers only "
            "(e.g., api-timeout)."
        )


def default_git_config_lookup(key: str) -> str | None:
    git_cmd = shutil.which("git")
    if not git_cmd:
        return None
    result = subprocess.run(  # noqa: S603
        [git_cmd, "config", key],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def default_env_lookup(name: str) -> str | None:
    value = os.getenv(name)
    return value if value and value.strip() else None


def get_author(
    git_lookup: Callable[[str], str | None] = default_git_config_lookup,
    env_lookup: Callable[[str], str | None] = default_env_lookup,
) -> str:
    author = git_lookup("user.name")
    if not author:
        author = env_lookup("USERNAME")
    if not author:
        return "Unknown"
    return author


def render_content(template: str, short_name: str, entry_date: str, author: str) -> str:
    updated = template.replace("<bug-name>", short_name)
    updated = updated.replace("YYYY-MM-DD", entry_date)
    updated = updated.replace("- Author: name", f"- Author: {author}")
    return updated


class FileSystem(Protocol):
    def ensure_dir(self, path: Path) -> None: ...

    def copy_file(self, src: Path, dest: Path) -> None: ...

    def read_text(self, path: Path) -> str: ...

    def write_text(self, path: Path, content: str) -> None: ...


@dataclass
class RealFileSystem(FileSystem):
    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def copy_file(self, src: Path, dest: Path) -> None:
        shutil.copyfile(src, dest)

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")


def default_code_launcher(files: Iterable[Path]) -> bool:
    code_cmd = shutil.which("code")
    if not code_cmd:
        return False
    subprocess.run([code_cmd, *[str(f) for f in files]], check=False)  # noqa: S603
    return True


def create_bug_entry(
    short_name: str,
    workspace: Path | None = None,
    fs: FileSystem | None = None,
    author_provider: Callable[[], str] = get_author,
    code_launcher: Callable[[Iterable[Path]], bool] = default_code_launcher,
    entry_date: str | None = None,
) -> Path:
    validate_short_name(short_name)

    workspace_path = workspace or _resolve_workspace()
    filesystem = fs or RealFileSystem()
    date_str = entry_date or date.today().strftime("%Y-%m-%d")

    target_dir = workspace_path / "docs" / "features" / "potential"
    target = target_dir / f"{date_str}-{short_name}.md"
    template = (
        workspace_path / "docs" / "features" / "templates" / "bug" / "potential_bug.md"
    )

    filesystem.ensure_dir(target_dir)
    filesystem.copy_file(template, target)

    author = author_provider()
    content = filesystem.read_text(target)
    updated = render_content(content, short_name, date_str, author)
    filesystem.write_text(target, updated)

    if not code_launcher([target]):
        print("WARNING: VS Code 'code' command not found. Open file manually:")
        print(f"  {target}")

    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a potential bug entry from the template."
    )
    parser.add_argument(
        "--short-name", required=True, help="Bug name in kebab-case (e.g., api-timeout)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        create_bug_entry(short_name=args.short_name)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        print(f"Aborted: required file not found: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

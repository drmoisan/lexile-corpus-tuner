"""Git helpers for PR context collection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import CommandResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class CommandRunner(Protocol):
    """Runs shell commands and returns structured results."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult: ...


class SubprocessRunner(CommandRunner):
    """Command runner that shells out using subprocess.run."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult:
        completed = subprocess.run(  # noqa: S603
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = (completed.stdout or "").rstrip("\n")
        stderr = (completed.stderr or "").rstrip("\n")
        result = CommandResult(
            stdout=stdout, stderr=stderr, code=int(completed.returncode)
        )

        if not allow_error and result.code != 0:
            joined = (stdout + "\n" + stderr).strip()
            raise RuntimeError(f"{' '.join(args)} failed ({result.code}): {joined}")

        return result


class GitClient:
    """Thin wrapper around git for typed access."""

    def __init__(self, runner: CommandRunner, cwd: Path) -> None:
        self._runner = runner
        self._cwd = cwd

    @property
    def cwd(self) -> Path:
        """Current working directory for git operations."""
        return self._cwd

    def run(self, args: Sequence[str], *, allow_error: bool = False) -> CommandResult:
        return self._runner.run(["git", *args], cwd=self._cwd, allow_error=allow_error)

    def resolve_root(self) -> Path:
        candidate = self._cwd / ".git"
        if candidate.exists():
            return self._cwd

        top = self.run(["rev-parse", "--show-toplevel"]).stdout
        # Preserve git's reported root without forcing OS-specific drive resolution
        return Path(top)

    def rev_parse(self, ref: str) -> str:
        return self.run(["rev-parse", "--verify", ref]).stdout

    def remote_verbose(self) -> str:
        return self.run(["remote", "-v"]).stdout

    def branch_name(self) -> str:
        return self.run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout

    def upstream(self) -> str:
        res = self.run(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            allow_error=True,
        )
        return res.stdout

    def status_short(self) -> str:
        return self.run(["status", "-sb"]).stdout

    def untracked(self) -> str:
        return self.run(["ls-files", "--others", "--exclude-standard"]).stdout

    def diff_name_status(self, *, staged: bool) -> str:
        args = ["diff", "--name-status"]
        if staged:
            args.insert(1, "--cached")
        return self.run(args, allow_error=True).stdout

    def diff_patch(self, *, staged: bool) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        return self.run(args, allow_error=True).stdout

    def merge_base(self, base: str, head: str) -> str:
        return self.run(["merge-base", base, head]).stdout

    def log(self, fmt: str, rev_range: str) -> str:
        return self.run(
            ["log", "--date=short", fmt, rev_range], allow_error=True
        ).stdout

    def diff_range(self, args: Sequence[str]) -> str:
        return self.run(["diff", *args], allow_error=True).stdout

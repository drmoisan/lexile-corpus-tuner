from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev_tools.pr_context_git import CommandResult, GitClient

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult:
        self.calls.append((tuple(args), allow_error))
        if tuple(args[:2]) == ("git", "rev-parse") and "--show-toplevel" in args:
            return CommandResult("/repo", "", 0)
        return CommandResult("value", "", 0)


def test_git_client_invokes_runner_commands(tmp_path: Path) -> None:
    runner = RecordingRunner()
    git = GitClient(runner, tmp_path)

    git.resolve_root()
    git.rev_parse("HEAD")
    git.remote_verbose()
    git.branch_name()
    git.upstream()
    git.status_short()
    git.untracked()
    git.diff_name_status(staged=True)
    git.diff_name_status(staged=False)
    git.diff_patch(staged=True)
    git.diff_patch(staged=False)
    git.merge_base("base", "head")
    git.log("--pretty=%s", "range")
    git.diff_range(["--stat"])

    called_commands = [call[0] for call in runner.calls]
    assert ("git", "rev-parse", "--show-toplevel") in called_commands
    assert ("git", "rev-parse", "--verify", "HEAD") in called_commands
    assert ("git", "remote", "-v") in called_commands
    assert ("git", "diff", "--stat") in called_commands

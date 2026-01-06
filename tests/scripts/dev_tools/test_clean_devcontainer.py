from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.dev_tools import clean_devcontainer as cleaner


class FakeRunner:
    def __init__(
        self, responses: dict[tuple[str, ...], list[cleaner.CommandResult]]
    ) -> None:
        self.responses = {key: list(values) for key, values in responses.items()}
        self.calls: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> cleaner.CommandResult:
        command_key = tuple(command)
        self.calls.append(list(command))
        if command_key not in self.responses or not self.responses[command_key]:
            raise AssertionError(f"No response configured for {command_key}")
        return self.responses[command_key].pop(0)


def make_result(
    returncode: int, *, stderr: str = "", stdout: str = ""
) -> cleaner.CommandResult:
    return cleaner.CommandResult(
        command=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_successful_cleanup_returns_zero_and_calls_all_commands() -> None:
    runner = FakeRunner(
        {
            ("docker", "rm", "-f", cleaner.DEFAULT_CONTAINER_NAME): [make_result(0)],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[0]): [make_result(0)],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[1]): [make_result(0)],
        }
    )

    exit_code = cleaner.clean_devcontainer(runner=runner)

    assert exit_code == 0
    assert runner.calls == [
        ["docker", "rm", "-f", cleaner.DEFAULT_CONTAINER_NAME],
        ["docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[0]],
        ["docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[1]],
    ]


def test_missing_resources_are_treated_as_success() -> None:
    runner = FakeRunner(
        {
            ("docker", "rm", "-f", cleaner.DEFAULT_CONTAINER_NAME): [
                make_result(1, stderr="Error: No such container")
            ],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[0]): [
                make_result(1, stderr="Error: No such volume")
            ],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[1]): [
                make_result(1, stderr="volume not found")
            ],
        }
    )

    exit_code = cleaner.clean_devcontainer(runner=runner)

    assert exit_code == 0
    assert len(runner.calls) == 3


def test_failure_propagates_nonzero_exit() -> None:
    runner = FakeRunner(
        {
            ("docker", "rm", "-f", cleaner.DEFAULT_CONTAINER_NAME): [make_result(0)],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[0]): [
                make_result(1, stderr="permission denied")
            ],
            ("docker", "volume", "rm", cleaner.DEFAULT_VOLUMES[1]): [make_result(0)],
        }
    )

    exit_code = cleaner.clean_devcontainer(runner=runner)

    assert exit_code == 1
    # All commands should still have been attempted.
    assert len(runner.calls) == 3


def test_run_command_executes_subprocess() -> None:
    # Use Python itself as a harmless command to avoid Docker dependencies.
    result = cleaner.run_command([sys.executable, "-c", "print('ok')"])

    assert result.returncode == 0
    assert "ok" in result.stdout

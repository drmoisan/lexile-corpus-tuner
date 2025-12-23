from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, cast

from scripts.dev_tools import fix_all

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pytest import MonkeyPatch


def make_result(code: int, output: str = "") -> fix_all.CommandResult:
    return fix_all.CommandResult(returncode=code, output=output)


class FakeRunner:
    def __init__(self, responses: dict[str, Iterable[fix_all.CommandResult]]) -> None:
        self.responses = {name: list(values) for name, values in responses.items()}
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, command: Sequence[str], *, step_name: str) -> fix_all.CommandResult:
        self.calls.append((step_name, list(command)))
        if step_name not in self.responses or not self.responses[step_name]:
            raise AssertionError(f"No response configured for {step_name}")
        return self.responses[step_name].pop(0)


def build_logger() -> fix_all.StepLogger:
    return fix_all.StepLogger(stream=StringIO())


def read_log(logger: fix_all.StepLogger) -> str:
    return cast(StringIO, logger.stream).getvalue()


def test_command_runner_accepts_stderr_when_exit_code_zero(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression: stderr output with exit code 0 should be treated as success."""
    captured = StringIO()
    logger = fix_all.StepLogger(stream=captured)

    def fake_run(
        command: Sequence[str], check: bool, capture_output: bool, text: bool
    ) -> object:  # type: ignore[override]
        assert command == ["demo"]
        assert not check
        assert capture_output
        assert text

        class Result:
            stdout = ""
            stderr = "All done! 98 files left unchanged.\n"
            returncode = 0

        return Result()

    monkeypatch.setattr(fix_all, "subprocess_run", fake_run)
    runner = fix_all.SubprocessCommandRunner(logger)
    result = runner.run(["demo"], step_name="Black: format")
    assert result.returncode == 0
    assert "98 files left unchanged" in captured.getvalue()


def test_command_runner_captures_output_on_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Command runner surfaces stdout/stderr even when the command fails."""
    captured = StringIO()
    logger = fix_all.StepLogger(stream=captured)

    def fake_run(
        command: Sequence[str], check: bool, capture_output: bool, text: bool
    ) -> object:  # type: ignore[override]
        assert command == ["demo"]
        assert not check
        assert capture_output
        assert text

        class Result:
            stdout = "line from stdout\n"
            stderr = "line from stderr\n"
            returncode = 1

        return Result()

    monkeypatch.setattr(fix_all, "subprocess_run", fake_run)
    runner = fix_all.SubprocessCommandRunner(logger)
    result = runner.run(["demo"], step_name="Black: format")
    assert result.returncode == 1
    logged = captured.getvalue()
    assert "line from stdout" in logged
    assert "line from stderr" in logged


def test_black_retries_before_success() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(1), make_result(1), make_result(0)],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(0)],
            "Pytest: test with coverage": [make_result(0)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_black_retries=3, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 0
    assert [call[0] for call in runner.calls].count("Black: format") == 3


def test_black_retries_exhausted() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(1), make_result(1), make_result(1)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_black_retries=3, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 1
    assert [call[0] for call in runner.calls] == [
        "JSON: format",
        "JSON: validate",
        "Black: format",
        "Black: format",
        "Black: format",
    ]
    assert "Black formatting failed after 3 attempts" in read_log(logger)


def test_pipeline_succeeds_when_black_writes_to_stderr() -> None:
    """Black stderr output without errors should not fail the pipeline."""
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(0, "stderr noise\n")],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(0)],
            "Pytest: test with coverage": [make_result(0)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 0
    assert [call[0] for call in runner.calls][-1] == "Pytest: test with coverage"


def test_pipeline_fails_when_black_fails() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [
                make_result(1, "error"),
                make_result(1, "error"),
                make_result(1, "error"),
            ],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=2, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 1
    assert [call[0] for call in runner.calls] == [
        "JSON: format",
        "JSON: validate",
        "Black: format",
        "Black: format",
        "Black: format",
    ]
    assert "Black formatting failed" in read_log(logger)


def test_ruff_retries_and_eventually_succeeds() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0), make_result(0)],
            "JSON: validate": [make_result(0), make_result(0)],
            "Black: format": [make_result(0), make_result(0)],
            "Ruff: lint": [make_result(1), make_result(0)],
            "Ruff: fix": [make_result(1), make_result(0)],
            "Pyright: type-check": [make_result(0)],
            "Pytest: test with coverage": [make_result(0)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=3, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 0
    assert [call[0] for call in runner.calls] == [
        "JSON: format",
        "JSON: validate",
        "Black: format",
        "Ruff: lint",
        "Ruff: fix",
        "Ruff: fix",
        "Black: format",
        "Ruff: lint",
        "Pyright: type-check",
        "Pytest: test with coverage",
    ]


def test_ruff_retries_exhausted() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(0)],
            "Ruff: lint": [make_result(1)],
            "Ruff: fix": [make_result(1), make_result(1)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=2, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 1
    assert [call[0] for call in runner.calls] == [
        "JSON: format",
        "JSON: validate",
        "Black: format",
        "Ruff: lint",
        "Ruff: fix",
        "Ruff: fix",
    ]
    assert "Ruff linting failed after 2 attempts" in read_log(logger)


def test_pipeline_runs_steps_in_order() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(0)],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(0)],
            "Pytest: test with coverage": [make_result(0)],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 0
    assert [call[0] for call in runner.calls] == [
        "JSON: format",
        "JSON: validate",
        "Black: format",
        "Ruff: lint",
        "Pyright: type-check",
        "Pytest: test with coverage",
    ]


def test_pipeline_stops_on_pyright_failure() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(0)],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(1, "type errors")],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 1
    assert [call[0] for call in runner.calls][-1] == "Pyright: type-check"
    assert "Pyright type checking failed" in read_log(logger)


def test_pipeline_stops_on_pytest_failure() -> None:
    runner = FakeRunner(
        {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
            "Black: format": [make_result(0)],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(0)],
            "Pytest: test with coverage": [make_result(1, "tests failed")],
        }
    )
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1, include_coverage=True, runner=runner, logger=logger
    )
    assert exit_code == 1
    assert [call[0] for call in runner.calls][-1] == "Pytest: test with coverage"
    assert "Pytest failed" in read_log(logger)

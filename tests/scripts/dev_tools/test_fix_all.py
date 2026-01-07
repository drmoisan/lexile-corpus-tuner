from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, cast

from scripts.dev_tools import fix_all

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pytest import MonkeyPatch


def make_result(code: int, output: str = "") -> fix_all.CommandResult:
    return fix_all.CommandResult(returncode=code, output=output)


class FakeRunner:
    def __init__(
        self, responses: Mapping[str, Iterable[fix_all.CommandResult]]
    ) -> None:
        self.responses = {name: list(values) for name, values in responses.items()}
        self.calls: list[tuple[str, list[str]]] = []
        self.branch_name: str | None = None

    def run(self, command: Sequence[str], *, step_name: str) -> fix_all.CommandResult:
        self.calls.append((step_name, list(command)))
        if step_name not in self.responses or not self.responses[step_name]:
            raise AssertionError(f"No response configured for {step_name}")
        return self.responses[step_name].pop(0)


class FakeRunnerFactory:
    def __init__(
        self,
        responses_by_branch: Mapping[
            str, Mapping[str, Iterable[fix_all.CommandResult]]
        ],
    ) -> None:
        self.responses_by_branch = responses_by_branch
        self.runners: dict[str, FakeRunner] = {}

    def __call__(
        self, branch_name: str, branch_logger: fix_all.StepLogger
    ) -> FakeRunner:
        runner = FakeRunner(self.responses_by_branch[branch_name])
        runner.branch_name = branch_name
        self.runners[branch_name] = runner
        return runner


def build_logger() -> fix_all.StepLogger:
    return fix_all.StepLogger(stream=StringIO())


def read_log(logger: fix_all.StepLogger) -> str:
    return cast(StringIO, logger.stream).getvalue()


def base_success_responses(
    *, include_coverage: bool = True
) -> dict[str, dict[str, list[fix_all.CommandResult]]]:
    pytest_key = "Pytest: test with coverage" if include_coverage else "Pytest: test"
    return {
        "json": {
            "JSON: format": [make_result(0)],
            "JSON: validate": [make_result(0)],
        },
        "shell": {
            "Shell: format": [make_result(0)],
            "Shell: check": [make_result(0)],
            "Shell: test": [make_result(0)],
        },
        "python": {
            "Black: format": [make_result(0)],
            "Ruff: lint": [make_result(0)],
            "Pyright: type-check": [make_result(0)],
            pytest_key: [make_result(0)],
        },
        "powershell": {
            "PoshQC: format": [make_result(0)],
            "PoshQC: analyze": [make_result(0)],
            "PoshQC: test": [make_result(0)],
        },
    }


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
    responses = base_success_responses()
    responses["python"]["Black: format"] = [
        make_result(1),
        make_result(1),
        make_result(0),
    ]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_black_retries=3,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 0
    python_calls = [call[0] for call in factory.runners["python"].calls]
    assert python_calls.count("Black: format") == 3


def test_black_retries_exhausted() -> None:
    responses = base_success_responses()
    responses["python"]["Black: format"] = [
        make_result(1),
        make_result(1),
        make_result(1),
    ]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_black_retries=3,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 1
    python_calls = [call[0] for call in factory.runners["python"].calls]
    assert python_calls == [
        "Black: format",
        "Black: format",
        "Black: format",
    ]
    assert "Black formatting failed after 3 attempts" in read_log(logger)


def test_pipeline_succeeds_when_black_writes_to_stderr() -> None:
    """Black stderr output without errors should not fail the pipeline."""
    responses = base_success_responses()
    responses["python"]["Black: format"] = [make_result(0, "stderr noise\n")]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 0
    assert [call[0] for call in factory.runners["powershell"].calls][
        -1
    ] == "PoshQC: test"


def test_pipeline_fails_when_black_fails() -> None:
    responses = base_success_responses()
    responses["python"]["Black: format"] = [
        make_result(1, "error"),
        make_result(1, "error"),
        make_result(1, "error"),
    ]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=2,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 1
    python_calls = [call[0] for call in factory.runners["python"].calls]
    assert python_calls == [
        "Black: format",
        "Black: format",
        "Black: format",
    ]
    assert "Black formatting failed" in read_log(logger)


def test_ruff_retries_and_eventually_succeeds() -> None:
    responses = base_success_responses()
    responses["python"]["Black: format"] = [make_result(0), make_result(0)]
    responses["python"]["Ruff: lint"] = [make_result(1), make_result(0)]
    responses["python"]["Ruff: fix"] = [make_result(1), make_result(0)]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=3, include_coverage=True, runner_factory=factory, logger=logger
    )
    assert exit_code == 0
    python_calls = [call[0] for call in factory.runners["python"].calls]
    assert python_calls == [
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
    responses = base_success_responses()
    responses["python"]["Ruff: lint"] = [make_result(1)]
    responses["python"]["Ruff: fix"] = [make_result(1), make_result(1)]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=2,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 1
    python_calls = [call[0] for call in factory.runners["python"].calls]
    assert python_calls == [
        "Black: format",
        "Ruff: lint",
        "Ruff: fix",
        "Ruff: fix",
    ]
    assert "Ruff linting failed after 2 attempts" in read_log(logger)


def test_pipeline_runs_steps_in_order() -> None:
    factory = FakeRunnerFactory(base_success_responses())
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 0
    assert [call[0] for call in factory.runners["json"].calls] == [
        "JSON: format",
        "JSON: validate",
    ]
    assert [call[0] for call in factory.runners["shell"].calls] == [
        "Shell: format",
        "Shell: check",
        "Shell: test",
    ]
    assert [call[0] for call in factory.runners["python"].calls] == [
        "Black: format",
        "Ruff: lint",
        "Pyright: type-check",
        "Pytest: test with coverage",
    ]
    assert [call[0] for call in factory.runners["powershell"].calls] == [
        "PoshQC: format",
        "PoshQC: analyze",
        "PoshQC: test",
    ]


def test_pipeline_stops_on_pyright_failure() -> None:
    responses = base_success_responses()
    responses["python"]["Pyright: type-check"] = [make_result(1, "type errors")]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 1
    assert [call[0] for call in factory.runners["python"].calls][
        -1
    ] == "Pyright: type-check"
    assert "Pyright type checking failed" in read_log(logger)


def test_pipeline_stops_on_pytest_failure() -> None:
    responses = base_success_responses()
    responses["python"]["Pytest: test with coverage"] = [make_result(1, "tests failed")]
    factory = FakeRunnerFactory(responses)
    logger = build_logger()
    exit_code = fix_all.run_fix_all(
        max_ruff_retries=1,
        include_coverage=True,
        runner_factory=factory,
        logger=logger,
    )
    assert exit_code == 1
    assert [call[0] for call in factory.runners["python"].calls][
        -1
    ] == "Pytest: test with coverage"
    assert "Pytest failed" in read_log(logger)

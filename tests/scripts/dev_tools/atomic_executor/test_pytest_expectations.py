"""
Tests for pytest expectation resolution and failure parsing helpers.
"""

from scripts.dev_tools.atomic_executor.plan_parser import PlanModel, PlanTask
from scripts.dev_tools.atomic_executor.pytest_expectations import (
    parse_pytest_failure_output,
    resolve_checked_test_expectations,
)


class TestResolveCheckedTestExpectations:
    """Tests for resolve_checked_test_expectations."""

    def test_expect_pass_overrides_expect_fail(self) -> None:
        """
        Expected-pass overrides expected-fail for the same test ref.

        Purpose:
            Ensure the override rule keeps a ref in expected-pass only.
        """
        test_ref = "tests/bugs/2026/test_issue_98.py::test_expected_fail"
        plan = PlanModel(
            tasks=[
                PlanTask(
                    "P1-T1",
                    1,
                    1,
                    "pytest tests/bugs/2026/test_issue_98.py::test_expected_fail",
                    True,
                    0,
                    expect_fail=True,
                    test_ref=test_ref,
                ),
                PlanTask(
                    "P1-T2",
                    1,
                    2,
                    "pytest tests/bugs/2026/test_issue_98.py::test_expected_fail",
                    True,
                    1,
                    expect_pass=True,
                    test_ref=test_ref,
                ),
            ],
            phases=[1],
        )

        expectations = resolve_checked_test_expectations(plan)

        assert expectations.expected_fail_refs == set()
        assert expectations.expected_pass_refs == {test_ref}
        assert expectations.missing_test_refs == []

    def test_only_checked_tasks_contribute_expectations(self) -> None:
        """
        Unchecked tasks should not contribute to expectation sets.

        Purpose:
            Guard against uncompleted work affecting QC behavior.
        """
        plan = PlanModel(
            tasks=[
                PlanTask(
                    "P1-T1",
                    1,
                    1,
                    "pytest tests/bugs/2026/test_issue_98.py::test_expected_fail",
                    False,
                    0,
                    expect_fail=True,
                    test_ref="tests/bugs/2026/test_issue_98.py::test_expected_fail",
                ),
                PlanTask(
                    "P1-T2",
                    1,
                    2,
                    "pytest tests/bugs/2026/test_issue_98.py::test_expected_pass",
                    True,
                    1,
                    expect_pass=True,
                    test_ref="tests/bugs/2026/test_issue_98.py::test_expected_pass",
                ),
            ],
            phases=[1],
        )

        expectations = resolve_checked_test_expectations(plan)

        assert expectations.expected_fail_refs == set()
        assert expectations.expected_pass_refs == {
            "tests/bugs/2026/test_issue_98.py::test_expected_pass"
        }
        assert expectations.missing_test_refs == []


class TestParsePytestFailureOutput:
    """Tests for parse_pytest_failure_output."""

    def test_parses_failing_nodeids(self) -> None:
        """
        Pytest failing nodeids should be extracted from summary output.

        Purpose:
            Ensure parameterized nodeids are captured verbatim.
        """
        output = "\n".join(
            [
                (
                    "========================= short test summary info "
                    "========================="
                ),
                (
                    "FAILED tests/bugs/2026/test_issue_98.py::"
                    "test_preflight_respects_expectations - AssertionError"
                ),
                (
                    "FAILED tests/bugs/2026/test_issue_98.py::"
                    "test_preflight_respects_expectations[param0] - AssertionError"
                ),
                (
                    "FAILED tests/other/test_other.py::test_unrelated "
                    "- AssertionError"
                ),
                (
                    "====================== 3 failed, 10 passed in 0.21s "
                    "======================"
                ),
            ]
        )
        summary = parse_pytest_failure_output(output)

        assert summary.failed_nodeids == {
            "tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations",
            "tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations[param0]",
            "tests/other/test_other.py::test_unrelated",
        }
        assert summary.has_collection_error is False

    def test_detects_collection_errors(self) -> None:
        """
        Collection/import errors should be treated as gate failures.

        Purpose:
            Ensure collection errors are flagged separately from test failures.
        """
        output = "\n".join(
            [
                (
                    "============================= test session starts "
                    "============================="
                ),
                "ERROR collecting tests/bugs/2026/test_issue_98.py",
                (
                    "ImportError while importing test module "
                    "'/workspaces/.../tests/bugs/2026/test_issue_98.py'."
                ),
                "E   ModuleNotFoundError: No module named 'some_missing_dep'",
                (
                    "=========================== short test summary info "
                    "============================"
                ),
                "ERROR tests/bugs/2026/test_issue_98.py",
                (
                    "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
                    "!!!!!!!!!!!!!!!!!!!!"
                ),
            ]
        )
        summary = parse_pytest_failure_output(output)

        assert summary.failed_nodeids == set()
        assert summary.has_collection_error is True

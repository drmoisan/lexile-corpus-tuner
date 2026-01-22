"""
Expectation resolution and pytest failure parsing helpers for atomic executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.dev_tools.atomic_executor.plan_parser import PlanModel

FAILED_NODEID_RE = re.compile(r"^FAILED\s+(?P<nodeid>\S+)")
COLLECTION_ERROR_RE = re.compile(r"^ERROR\s+collecting\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ResolvedTestExpectations:
    """
    Resolved expectations for pytest outcomes derived from a plan.

    Purpose:
        Provide normalized expectation sets for QC gating decisions.

    Attributes:
        expected_fail_refs (set[str]): Test refs allowed to fail.
        expected_pass_refs (set[str]): Test refs required to pass.
        missing_test_refs (list[str]): Checked expectation task IDs lacking test refs.
    """

    expected_fail_refs: set[str]
    expected_pass_refs: set[str]
    missing_test_refs: list[str]


@dataclass(frozen=True)
class PytestFailureSummary:
    """
    Parsed summary of pytest failures from captured output.

    Purpose:
        Normalize pytest output into data used for QC gate decisions.

    Attributes:
        failed_nodeids (set[str]): Nodeids reported as failed.
        has_collection_error (bool): True when collection/import errors appear.
    """

    failed_nodeids: set[str]
    has_collection_error: bool


def resolve_checked_test_expectations(plan: PlanModel) -> ResolvedTestExpectations:
    """
    Resolve expectation tags from checked plan tasks.

    Purpose:
        Collect expected-fail and expected-pass refs derived from checked tasks.

    Args:
        plan (PlanModel): Parsed plan model to inspect.

    Returns:
        ResolvedTestExpectations: Normalized expectation sets and missing refs.

    """
    expected_fail_refs: set[str] = set()
    expected_pass_refs: set[str] = set()
    missing_test_refs: list[str] = []

    # Walk checked tasks with expectation tags and gather their test references.
    for task in plan.tasks:
        if not task.checked:
            continue
        if not (task.expect_fail or task.expect_pass):
            continue
        if not task.test_ref:
            missing_test_refs.append(task.task_id)
            continue
        if task.expect_pass:
            expected_pass_refs.add(task.test_ref)
        elif task.expect_fail:
            expected_fail_refs.add(task.test_ref)

    # Enforce override semantics so expected-pass wins for the same ref.
    expected_fail_refs.difference_update(expected_pass_refs)

    return ResolvedTestExpectations(
        expected_fail_refs=expected_fail_refs,
        expected_pass_refs=expected_pass_refs,
        missing_test_refs=missing_test_refs,
    )


def parse_pytest_failure_output(output: str) -> PytestFailureSummary:
    """
    Parse pytest output into failing nodeids and collection status.

    Purpose:
        Identify failing tests and collection/import errors from pytest output.

    Args:
        output (str): Combined stdout/stderr captured from pytest.

    Returns:
        PytestFailureSummary: Parsed nodeids and collection error flag.

    """
    failed_nodeids: set[str] = set()
    has_collection_error = False

    lines = output.splitlines()

    # Scan pytest output line-by-line to extract failures and error conditions.
    for line in lines:
        stripped = line.strip()
        failed_match = FAILED_NODEID_RE.match(stripped)
        if failed_match:
            failed_nodeids.add(failed_match.group("nodeid"))
            continue

        # Flag collection/import errors so QC gates can fail fast.
        if COLLECTION_ERROR_RE.match(stripped):
            has_collection_error = True
            continue
        if stripped.lower().startswith("importerror while importing test module"):
            has_collection_error = True
            continue
        if stripped.lower().startswith("error during collection"):
            has_collection_error = True
            continue
        if stripped.lower().startswith("error "):
            has_collection_error = True

    return PytestFailureSummary(
        failed_nodeids=failed_nodeids,
        has_collection_error=has_collection_error,
    )

# 2026-01-20-atomic-executor-qc-regression-test (Spec)

- **Issue:** #98
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T10-11
- **Status:** Draft
- **Version:** 0.1

## Context
When the atomic executor exits mid-execution after completing TDD "red" regression tests (tasks tagged with `[expect-fail]`), restarting the executor causes pre-flight QC to fail on those intentionally-failing tests. Pre-flight QC is unaware of the plan's `[expect-fail]` semantics.

Environment:
- OS/version: Linux (devcontainer)
- Python version: 3.13
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all <plan> --workspace <path>`
- Data source or fixture: Any plan with `[expect-fail]` TDD tasks in an early phase

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

The issue only manifests when the executor is interrupted mid-TDD-workflow and restarted. Workaround exists.


## Repro & Evidence
Steps to Reproduce:
1. Create an atomic plan with Phase 1 TDD regression tests tagged with `[expect-fail]`
2. Run the atomic executor (`execute-all`)
3. Allow Phase 1 `[expect-fail]` tasks to complete (tests are created and fail as expected)
4. Interrupt or crash the executor before Phase 2 implementation tasks complete
5. Restart the executor on the same plan (without `--skip-preflight-qc`)
6. Pre-flight QC runs pytest, detects the failing regression tests, and attempts to "fix" them

Expected:
Pre-flight QC should recognize that failing tests correspond to completed `[expect-fail]` plan tasks with pending implementation tasks, and either:
- Skip those specific tests during pre-flight QC, or
- Skip pre-flight QC entirely when resuming mid-plan execution

Actual:
Pre-flight QC treats the failing regression tests as unexpected failures and invokes Copilot to fix them, which defeats the TDD workflow.

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet: (observed behavior, no specific log captured)


## Scope & Non-Goals
- In scope:
- In scope:
	- Make QC gates plan-aware for intentional TDD “red” failures created by checked `[expect-fail]` tasks.
		- Applies to:
			- Pre-flight QC run by the atomic executor CLI (unless `--skip-preflight-qc`).
			- Phase-end QC (when the executor runs QC at phase boundaries).
	- Extend plan parsing semantics to support plan-linked expectations that can be evaluated deterministically:
		- `[expect-fail]` tasks must include an explicit test reference (a pytest nodeid or nodeid prefix).
		- Add `[expect-pass]` as an explicit override that requires the referenced test(s) to pass.
	- Ensure pre-flight QC does not enter the “fix baseline QC” loop when failures are explicitly expected.

- Out of scope / non-goals:
	- Disabling QC entirely by default (the goal is to keep QC running while ignoring only explicitly expected failures).
	- Heuristic inference of “which failing tests correspond to which plan tasks” without explicit test references.
	- Converting tests to use pytest `xfail` markers as the primary mechanism (expectations should live in the plan).
	- Introducing a separate persistent state file (for example `.expect_fail_tests.json`) as the source of truth.

- Explicitly excluded systems, integrations, or datasets:
	- No external services.
	- No datasets beyond local plan files and normal repo test suite execution.

## Root Cause Analysis
The `[expect-fail]` tag lives in the plan file and is only interpreted during task execution. Pre-flight QC is currently plan-agnostic and assumes “pytest must pass”, so it treats intentionally failing TDD-red tests as unexpected failures and invokes the “fix baseline QC” loop.

Note:
- Pre-flight QC runs after plan parsing/validation (it is not blocked on lack of plan parsing); the issue is that QC does not apply plan semantics when interpreting pytest failures.

Relevant files:
- `scripts/dev_tools/atomic_executor/cli.py` — pre-flight QC logic
- `scripts/dev_tools/atomic_executor/plan_parser.py` — `[expect-fail]` parsing


## Proposed Fix

### Design summary (what changes where):

Implement plan-linked expected failures so the executor can run QC (including pytest) while ignoring only failures that are explicitly marked as expected by checked plan tasks.

Concretely:
- Extend plan semantics:
	- Support `[expect-pass]` (in addition to existing `[expect-fail]`).
	- Require a machine-resolvable test reference (`test_ref`) on expectation-tagged tasks.
		- Preferred syntax: a `pytest <nodeid>` prefix.
		- Backward-compatible syntax: parse the existing prose pattern `Add pytest `test_name` in `tests/path/test_file.py`` into a nodeid prefix.
- Make QC interpretation plan-aware:
	- Run pytest as usual, but treat non-zero exit as a gate failure only when at least one failing test is unexpected.
	- Expectation precedence: if the same `test_ref` is both expected-fail and expected-pass (because both tasks are checked), expected-pass wins.
	- Collection/import errors are always unexpected gate failures (no stable nodeids to match, and they indicate broken test environment).

Where this lives:
- Plan parsing changes in `scripts/dev_tools/atomic_executor/plan_parser.py`.
- QC gate interpretation changes in:
	- `scripts/dev_tools/atomic_executor/cli.py` (pre-flight QC path).
	- `scripts/dev_tools/atomic_executor/qc_runner.py` (phase-end QC path), so behavior is consistent.

### Boundaries and invariants to preserve:

- The existing plan checkbox semantics (`checked` tasks) remain the single source of truth for “completed work”.
- The existing `[expect-fail]` parsing behavior remains supported (tag stripped from the human-readable task title; stored as semantic state).
- `--skip-preflight-qc` remains a supported workaround/escape hatch.
- QC still runs the full toolchain (Black → Ruff → Pyright → Pytest) unless explicitly skipped; only the interpretation of pytest failures changes.

### Dependencies or blocked work:

- None (no new external services/libraries required).

### Implementation strategy (what changes, not sequencing):
	
#### Files/modules to change:

- `scripts/dev_tools/atomic_executor/plan_parser.py`
	- Add `[expect-pass]` parsing.
	- Extract `test_ref` from expectation-tagged tasks using a deterministic grammar.
- `scripts/dev_tools/atomic_executor/cli.py`
	- Pre-flight QC: incorporate plan-linked expectations when evaluating pytest failures.
	- Avoid entering the “fix baseline QC” loop when pytest failures are expected.
- `scripts/dev_tools/atomic_executor/qc_runner.py`
	- Phase-end QC: apply the same expectation filtering so phase gates behave the same as pre-flight.

Optional (recommended for cohesion):
- Add a small, pure helper module under `scripts/dev_tools/atomic_executor/` to host:
	- Expectation resolution from a parsed plan.
	- Pytest failing nodeid parsing.
	- Gate decision logic.

#### Functions/classes/CLI commands impacted:

- `scripts/dev_tools/atomic_executor/cli.py`
	- `_run_preflight_qc_with_capture(...)` (pytest step output capture and gate decision)
	- `_run_preflight_qc_fix_loop(...)` (do not treat expected failures as “baseline QC needs fixing”)
- `scripts/dev_tools/atomic_executor/plan_parser.py`
	- `PlanTask` parsing logic for `expect_fail` and the new `[expect-pass]` semantics.
- `scripts/dev_tools/atomic_executor/qc_runner.py`
	- Full QC runner used by the executor at phase boundaries.

#### Data flow and validation changes:

Plan → expectations → QC gate:

- Plan parsing produces tasks with expectation metadata.
- Resolve expectations from checked tasks only:
	- `expected_fail_refs: set[str]`
	- `expected_pass_refs: set[str]`
	- Override rule: if a `test_ref` is present in both sets, treat it as expected-pass.

Validation:
- Any expectation-tagged task without a machine-resolvable `test_ref` is a plan authoring error for plan-linked gating.
- Selected enforcement behavior:
	- In pre-flight QC, fail fast if any checked expectation `test_ref` cannot be resolved to a collected test.
		- Implement via a probe collect step (for example `pytest --collect-only --color=no <ref...>`).
	- Do not require the referenced test to exist at plan authoring time; only enforce resolvability once the expectation task is checked.

Pytest output parsing:
- Run pytest with `--color=no` during QC to simplify parsing.
- Parse failing nodeids from pytest output (for example `FAILED path/to/test.py::test_name ...`).
- Match expectations via prefix match on nodeids, so parameterized tests (`...::test[param]`) are covered.

#### Error handling and logging updates:

- When an expectation-tagged task is checked but its `test_ref` cannot be collected:
	- Fail the QC gate with a clear, actionable message that includes the missing `test_ref` and points the user back to the plan line.
- When pytest fails with collection/import errors:
	- Treat as unexpected and preserve the captured output for diagnosis.
- Ensure logs clearly distinguish:
	- “Expected failures present; QC gate allowed to pass”
	- “Unexpected failures present; QC gate failed”

#### Rollback/feature-flag considerations (if applicable):

- No feature flag planned.
- Rollback is a simple revert of the plan-linked gating behavior to the current strict “pytest must pass” gate.

### Technical specifications (interfaces/contracts):

#### Inputs/outputs and formats:

Plan task title (interface):
- New: `[expect-pass]` tag in task titles.
- Existing: `[expect-fail]` tag.
- Expectation-tagged tasks must include a test reference.

Supported test reference forms (in order):
- `pytest <nodeid-or-prefix>`
	- Example: `pytest tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations`
- `Add pytest `test_name` in `tests/path/test_file.py``
	- Interpreted as `<tests/path/test_file.py>::<test_name>`

QC output parsing input:
- Captured stdout/stderr from pytest invocation.

QC gate decision output:
- A boolean “gate passes” decision plus captured tool output for reporting.

#### Required configuration keys and defaults:

- None.

Notes:
- This design intentionally avoids persistent state files; the plan is the source of truth.

#### Backward-compatibility expectations:

- Plans without expectation tags behave as they do today (pytest must pass).
- Existing `[expect-fail]` tagged tasks continue to parse, but plan-linked gating requires a resolvable `test_ref` once the task is checked.
- Plans that include expectation tags without resolvable test references should fail plan validation (or pre-flight QC) with an actionable error.

#### Performance constraints (latency/throughput/memory):

- Pre-flight QC remains dominated by the existing full toolchain.
- The additional “collect-only” probe for checked expectation refs should be small relative to a full pytest run.
- Pytest output parsing is linear in captured output size.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
- Assumptions (environment, data, access):
	- Executor runs in a Poetry-managed environment and invokes pytest via `poetry run pytest ...`.
	- Plan parsing already occurs before the executor decides whether to run pre-flight QC.
	- Expectation-tagged tasks are only “authoritative” when checked.

- Constraints (budget, performance, compatibility):
	- Must remain deterministic and unit-testable without touching the filesystem or network.
	- Must preserve repo QC standards (Black/Ruff/Pyright/Pytest toolchain).
	- Must work cross-platform (Linux devcontainer is primary repro; behavior should not assume OS-specific paths beyond plan nodeids).

- External dependencies (services, libraries, releases):
	- None.

## Data / API / Config Impact
- User-facing or API changes:
- User-facing or API changes:
	- Plan syntax becomes more expressive:
		- New `[expect-pass]` tag.
		- Expectation-tagged tasks must carry an explicit test reference (pytest nodeid/prefix).

- Data or migration considerations:
	- Existing plans that use `[expect-fail]` without a machine-resolvable test reference will need an update if they want plan-linked gating (add `pytest ...` or use the supported “Add pytest … in …” form).

- Logging/telemetry updates (if any):
	- Add explicit messaging in QC output to show which failing tests were treated as expected and why.

- Compatibility notes (CLI flags, config schemas, versioning):
	- No new CLI flags required.
	- `--skip-preflight-qc` remains supported.

## Test Strategy
Seeded from issue:

**Immediate workaround:**
- Use `--skip-preflight-qc` when resuming after interruption

**Potential fixes (future enhancement):**

| Approach | Pros | Cons |
|----------|------|------|
| **Plan-aware pre-flight** | Single source of truth | Complex: must parse plan, correlate tests |
| **State file** (`.expect_fail_tests.json`) | Pre-flight can exclude specific tests | State management, can get stale |
| **Auto-skip on resume** | Simple UX | May mask real issues |
| **Pytest `xfail` marker** | Standard pytest mechanism | Must remove marker after fix |

Recommended approach: **Plan-aware pre-flight QC** that:
1. Parses the plan before running QC
2. Finds `[expect-fail]` tasks that are checked (test created) but whose implementation sibling is unchecked
3. Runs pytest with `--deselect` for those specific test functions

- [ ] Unit coverage: test that pre-flight QC respects `[expect-fail]` completed tasks
- [ ] Integration scenario: interrupt executor after `[expect-fail]` task, resume, verify no spurious fix attempts
- [ ] Manual verification: use `--skip-preflight-qc` workaround

Updated recommendation (research): implement plan-linked expected failures instead of skipping/deselecting tests.

- Parser-level tests (fast, deterministic):
	- Extend existing plan parser tests to cover:
		- `[expect-pass]` parsing.
		- `test_ref` extraction from `pytest <nodeid>` and “Add pytest … in …” forms.
		- Tag stripping from the stored human title.

- Expectation resolution tests (pure logic):
	- Verify checked expectation refs are collected.
	- Verify override behavior: checked expected-pass overrides checked expected-fail for the same `test_ref`.

- Pytest output parsing tests (pure logic):
	- Parse failing nodeids from representative pytest outputs.
	- Validate prefix matching behavior for parameterized tests.
	- Treat collection/import errors as unexpected failures.

- Pre-flight QC gate integration tests (unit-level via mocks):
	- Patch subprocess calls to return controlled pytest output.
	- Assert gate passes when all failing nodeids match checked expected-fail refs and no overriding expected-pass exists.
	- Assert gate fails when an overriding expected-pass exists for a failing nodeid.
	- Assert gate fails when any failing nodeid is not covered by checked expected-fail refs.

- Regression tests to add or update:
- Unit tests (pytest) for the fixed behavior and boundaries:
- Edge cases and negative scenarios (invalid inputs, missing data, boundary values):
- Error handling and logging verification:
- Coverage impact and targets for changed lines/modules:
- Toolchain commands to run (format → lint → type-check → test):
- Manual validation steps (if required):


## Acceptance Criteria
- [ ] Repro steps now produce the expected behavior in all documented environments:
	- When resuming mid-plan after completing checked `[expect-fail]` “red” tasks, pre-flight QC does not treat the corresponding test failures as baseline QC failures.
	- QC still runs; only explicitly expected failures are ignored.
- [ ] Regression tests added and passing that cover the gate decision logic:
	- Unit tests for plan expectation parsing (`[expect-fail]` and `[expect-pass]`) and `test_ref` extraction.
	- Unit tests for pytest failing nodeid parsing and prefix matching.
	- Unit tests for pre-flight QC behavior in `TestPreflightQC` that assert expected failures do not trigger the fix loop.
- [ ] Edge cases and invalid inputs are handled with correct errors or fallbacks:
	- Checked expectation-tagged task without a resolvable `test_ref` fails fast with an actionable message.
	- Checked `test_ref` that cannot be collected by pytest fails fast (actionable).
	- Pytest collection/import errors are treated as unexpected failures (gate fails).
- [ ] No unintended behavior changes outside the defined scope:
	- Plans without expectation tags continue to require pytest to pass.
	- Non-matching failures remain gate failures.
- [ ] Required logs/telemetry updated and validated (if applicable):
	- QC output indicates whether failures were treated as expected and which expectation entries applied.
- [ ] Performance constraints met or explicitly waived with rationale:
	- The additional expectation resolution/probe step is small compared to a full pytest run.
- [ ] Full toolchain pass completed (format → lint → type-check → test).
- [ ] Docs/config references updated to match the new behavior:
	- Plan authoring guidance documents the `pytest <nodeid>` (or compatible) expectation syntax.

## Risks & Mitigations
- Technical or operational risks:
- Technical or operational risks:
	- Pytest output parsing can be brittle across output formats.
	- Plan authors may forget to include a `test_ref` on expectation-tagged tasks, leading to confusion.
	- Overly broad `test_ref` prefixes could accidentally “cover” unrelated failures.

- Mitigations and rollbacks:
	- Run pytest with `--color=no` in QC paths to reduce output variability and stabilize parsing.
	- Keep parsing logic isolated and fully unit-tested against representative output shapes.
	- Fail fast with actionable errors when checked expectation tasks are missing `test_ref` or cannot be collected.
	- Encourage precise nodeid prefixes (file + test function name) and document matching semantics.
	- Roll back by reverting to strict QC gating (pytest failures always fail) if needed.

## Rollout & Follow-up
- Release/rollout steps:
- Release/rollout steps:
	- Update any active plans using `[expect-fail]` to include explicit test references where plan-linked gating is desired.
	- If there is a standard plan template for atomic executor work, add a short note about expectation syntax.

- Post-fix monitoring or clean-up tasks:
	- Verify a real resume scenario: interrupt after a checked `[expect-fail]` “red” task, resume, confirm no spurious “fix baseline QC” actions.

- Links: issue, PRs, related docs
	- Issue: #98

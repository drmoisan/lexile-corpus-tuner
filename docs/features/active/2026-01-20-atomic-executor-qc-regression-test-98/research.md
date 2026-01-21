<!-- markdownlint-disable-file -->

# Task Research Notes: atomic-executor-preflight-qc-regression-98

## Research Executed

### File Analysis

- docs/features/active/2026-01-20-atomic-executor-qc-regression-test-98/spec.md
  - Verified the stated repro, acceptance options, and the spec’s recommended direction (plan-aware preflight; skip tests or skip preflight when resuming).
  - Confirmed the spec currently claims preflight runs “before plan parsing”, which is not accurate in the current `cli.py` (plan is parsed/validated before preflight QC runs).
- docs/features/active/2026-01-20-atomic-executor-qc-regression-test-98/plan.2026-01-21T09-06.md
  - Verified that Phase 2 contains `[expect-fail]` tasks but does **not** include any explicit pytest nodeids / test paths.
  - This matches the original spec’s “skip tests/skip preflight” framing, but does not support the newer requirement (plan-linked expected failures).
- scripts/dev_tools/atomic_executor/cli.py
  - Verified preflight QC always runs (unless `--skip-preflight-qc`) and uses a Copilot “fix baseline QC” loop on failure.
  - Verified plan parsing/validation happens before preflight QC is invoked, so plan-aware behavior can be implemented without re-architecting the CLI.
  - Verified preflight QC uses `_run_preflight_qc_with_capture()` which hardcodes a full toolchain including a full pytest invocation with coverage.
- scripts/dev_tools/atomic_executor/plan_parser.py
  - Verified `[expect-fail]` tag is already parsed into `PlanTask.expect_fail` and stripped from the stored task title.
  - The parser does not currently support `[expect-pass]` or any explicit “test reference” field for linking expectations to pytest results.
  - Verified `next_unchecked_task()` and `phase_complete()` are plan-state queries that can support resume-aware decisions.
- scripts/dev_tools/atomic_executor/qc_runner.py
  - Verified full/scoped QC orchestration exists but preflight QC currently uses its own subprocess implementation.
- tests/scripts/dev_tools/atomic_executor/test_cli.py
  - Verified `TestPreflightQC` already covers `_run_preflight_qc_with_capture()` success/failure and parsing of `--skip-preflight-qc`.

### Code Search Results

- "_run_preflight_qc_with_capture"
  - Found in `scripts/dev_tools/atomic_executor/cli.py` (definition and used by `_run_preflight_qc_fix_loop`).
- "skip_preflight_qc"
  - Found in `scripts/dev_tools/atomic_executor/cli.py` (arg + conditional) and in `tests/scripts/dev_tools/atomic_executor/test_cli.py` (parsing tests).
- "expect_fail"
  - Found in `scripts/dev_tools/atomic_executor/plan_parser.py` (parsed into `PlanTask.expect_fail`).

- "expect-pass"
  - No matches found in `scripts/dev_tools/atomic_executor/plan_parser.py` or `scripts/dev_tools/atomic_executor/cli.py` (as of this research snapshot).

### External Research

- #githubRepo:"(none)"
  - No GitHub repository searches were executed for this task.
- #fetch:(none)
  - No external URLs were fetched for this task.

### Project Conventions

- Standards referenced: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`.
- Instructions followed: Task Researcher mode (research-only; artifact written under `artifacts/research/` only).

## Key Discoveries

### Project Structure

- The atomic executor’s preflight QC path is in `scripts/dev_tools/atomic_executor/cli.py`:
  - `main()` parses and validates the plan via `PlanParser(plan_path)` and `parser.preflight_validate()`.
  - Then (unless `--skip-preflight-qc`) it runs preflight QC via `_run_preflight_qc_fix_loop()`.
  - `_run_preflight_qc_fix_loop()` uses `_run_preflight_qc_with_capture()` which executes Black → Ruff → Pyright → Pytest; on failure it invokes Copilot to “fix baseline QC”.

Evidence (from `cli.py`): preflight happens **after** plan parsing/validation.

### Implementation Patterns

- **Plan semantics already exist**: `PlanParser.parse()` sets `PlanTask.expect_fail` when a task title begins with `[expect-fail]`, and strips the tag from `PlanTask.title`.
- **Preflight QC is currently plan-agnostic**: preflight always assumes “pytest must pass”, so it will fail (and try to “fix baseline”) whenever the repo has intentionally failing tests after a completed `[expect-fail]` “red” task.

### Complete Examples

```python
# scripts/dev_tools/atomic_executor/cli.py (preflight QC runner)
steps = [
    ("black", ["poetry", "run", "black", "--check", "."]),
    ("ruff", ["poetry", "run", "ruff", "check"]),
    ("pyright", ["poetry", "run", "pyright"]),
    (
        "pytest",
        [
            "poetry",
            "run",
            "pytest",
            "--cov=src/lexile_corpus_tuner",
            "--cov=scripts/dev_tools",
            "--cov-report=term-missing",
        ],
    ),
]
```

```python
# scripts/dev_tools/atomic_executor/plan_parser.py ([expect-fail] parsing)
expect_fail = False
if raw_title.lower().startswith("[expect-fail]"):
    expect_fail = True
    raw_title = raw_title[len("[expect-fail]") :].strip()
...
PlanTask(..., expect_fail=expect_fail)
```

### API and Schema Documentation

- `PlanTask` includes:
  - `checked: bool` (checkbox state)
  - `expect_fail: bool` (TDD red marker)

### Configuration Examples

```text
(none)
```

### Technical Requirements

Derived from Issue #98 + the updated requirement (plan-linked expected failures):

1. **Plan-linked expectations**:
  - Plan tasks must be able to explicitly name a test (pytest nodeid or a stable prefix).
  - A checked `[expect-fail]` entry means: if that test fails, it is an *expected* failure (do not fail the gate).
  - If a checked `[expect-pass]` exists for the same test, that overrides `[expect-fail]` and the test must pass.

2. **QC should still run**:
  - Pre-flight QC and phase-end QC should still run pytest.
  - Gate failure should be based on *unexpected* failures after filtering expected failures.

3. **Determinism and safety**:
  - Only failures explicitly declared expected by checked plan tasks are ignored.
  - All other failures behave like normal QC failures.

**Mandatory unachievable objective callout**:
- **This approach is not achievable without some plan-level “test reference” convention** (e.g., pytest nodeids) because the executor cannot infer which failing tests correspond to which plan tasks. The current Issue #98 plan template does not include those references; the plan (and parser) must be extended.

## Recommended Approach

Implement **plan-linked expected failures** so QC still runs, but only *explicitly expected* failures are ignored.

### Selected approach: plan-linked `[expect-fail]` + `[expect-pass]` using explicit pytest nodeids

High-level behavior (applies to pre-flight QC and phase-end QC):

1. Parse the plan and collect **checked** expectation entries:
   - `[expect-fail] <test-ref>`
   - `[expect-pass] <test-ref>`
2. Run pytest normally (full suite).
3. If pytest exits non-zero, parse the failing test nodeids from pytest output.
4. Treat pytest as a gate failure only if there exists at least one **unexpected** failing test:
   - “Unexpected failing” means:
     - It does *not* match any checked `[expect-fail] <test-ref>` entries, OR
     - It matches a checked `[expect-pass] <test-ref>` entry (pass expectation overrides fail expectation).

This preserves the TDD workflow:
- Early “red” step: checked `[expect-fail]` entry exists and tests are allowed to fail.
- After fix step: checked `[expect-pass]` entry exists (same test-ref), making failures unacceptable.

#### Recommended plan syntax (minimal, human-friendly)

Evidence-based observation:

- Some existing plans already include a human-friendly “Add pytest … in …” convention.
  (Example called out in the conversation history: `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/plan.2026-01-20T16-24.md`.)

For plan-linked QC gating, the plan needs a machine-resolvable test reference.

Important clarification (answers “does the test need to exist to have a nodeid?”):

- A pytest “nodeid” is just a **string identifier convention** (typically
  `relative/path/to/test_file.py::test_function_name[...]`).
- You can write the canonical `pytest <nodeid>` form **before the test exists**, because
  it’s describing the *intended* test you plan to create.
- The test must exist by the time QC runs (otherwise pytest will fail due to collection/
  selection errors), but plan authoring can still use nodeids as a forward reference.

Recommended canonical form (most deterministic):

- `- [ ] [P2-T1] [expect-fail] pytest tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations`
- `- [ ] [P3-T1] [expect-pass] pytest tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations`

Backward-compatible accepted form (matches the existing plan-writing style):

- `- [ ] [P?-T?] [expect-fail] Add pytest `test_name` in `tests/path/test_file.py``
  - Interpreted as `test_ref = tests/path/test_file.py::test_name`.

Matching rules:

- Treat `test_ref` as a **pytest nodeid prefix**.
  - This makes parameterized tests work naturally: `path::test_name` matches
    `path::test_name[param0]`.
- If only a bare test function name is present (no file path), matching is ambiguous;
  avoid supporting it.

### Rejected alternatives (brief)

- **Skip preflight or skip pytest**: rejected because it avoids the core requirement (QC must still run; only specific failures are treated as expected).
- **Pytest xfail markers**: rejected because it couples expectation state to test code rather than the plan (and requires later cleanup/migration).
- **State file (.expect_fail_tests.json)**: rejected for added lifecycle complexity and staleness risk; the plan is the desired single source of truth.
- **Heuristic mapping from plan text to failing tests** (no nodeid convention): rejected as non-deterministic.

## Implementation Guidance

- **Objectives**:
  - Keep running QC (including pytest) while preventing “baseline fix” loops that would incorrectly try to eliminate intended TDD-red failures.
  - Make the plan the single source of truth for which failures are temporarily acceptable and when they must become unacceptable.

- **Key Tasks**:
  - Extend plan parsing to support expectations with explicit test references:
    - Add support for `[expect-pass]` tag.
    - Add support for extracting a `test_ref` (pytest nodeid or prefix) from the remainder of the title line.
    - Represent expectation in `PlanTask` as:
      - `expectation: Literal["none", "fail", "pass"]` (or two booleans), and
      - `test_ref: str | None`.

    - Parsing grammar recommendation (designed to be compatible with existing plan prose):
      - After stripping the leading `[expect-fail]` / `[expect-pass]` tag, attempt extraction in order:
        1) If the remaining title begins with `pytest ` (literal), treat the remainder as
           `test_ref` (trim whitespace).
           - Example: `pytest tests/x/test_y.py::test_z`
        2) Else, parse the “Add pytest … in …” pattern using backticks:
           - Conceptual regex: `Add pytest\s+`(?P<test_name>[^`]+)`\s+in\s+`(?P<path>tests/[^`]+)`
           - Build `test_ref = {path}::{test_name}`.
        3) Otherwise, no `test_ref` is present.

      - Validation policy:
        - An expectation-tagged task that lacks a machine-resolvable `test_ref` should be treated
          as a plan authoring error for plan-linked gating.
        - Recommended behavior: fail fast in `parser.preflight_validate()` with a clear message.
        - By contrast, **do not require** that `test_ref` resolves to an existing collected test
          during plan parsing. That would break the normal workflow where the plan is written
          *before* the test file/function exists.
          - However, once an expectation task is **checked**, it is asserting that the test now
            exists and can be referenced deterministically.

        - Selected enforcement policy (per user decision):
          - In **pre-flight QC**, fail fast if any **checked** expectation `test_ref` cannot be
            resolved to a collected test.
          - Implement as a dedicated resolution step before the full pytest run:
            - Resolve only checked expectation refs (both expect-fail and expect-pass).
            - Run a probe like `pytest --collect-only --color=no <ref1> <ref2> ...`.
            - If the probe exits non-zero or reports missing tests, treat it as a preflight gate
              failure with an actionable message pointing to the exact missing `test_ref`.

  - Build an expectation resolver that reads the *checked* tasks and produces two sets:
    - `expected_fail_refs: set[str]`
    - `expected_pass_refs: set[str]`
    - Resolution rule: if a test ref appears in both (because both steps are checked), treat it as **expected-pass**.

  - Run pytest and parse failing nodeids from output:
    - Prefer adding `--color=no` in the pytest invocation to simplify parsing.
    - Parse nodeids from lines starting with `FAILED ` (typical `-q` / default output) and/or summary blocks.
    - Matching rule: treat a `test_ref` as a **prefix match** on failing nodeids to support parametrized tests.

     - Representative pytest output shapes to cover in unit tests (no real pytest run):

      1) Short summary contains nodeids:

        ```text
        ========================= short test summary info =========================
        FAILED tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations - AssertionError: boom
        FAILED tests/other/test_other.py::test_unrelated - AssertionError: nope
        ======================= 2 failed, 10 passed in 0.21s =======================
        ```

      2) Parameterized nodeids:

        ```text
        ========================= short test summary info =========================
        FAILED tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations[param0] - AssertionError
        FAILED tests/bugs/2026/test_issue_98.py::test_preflight_respects_expectations[param1] - AssertionError
        ======================= 2 failed, 10 passed in 0.21s =======================
        ```

      3) Collection/import error (no stable nodeid):

        ```text
        ============================= test session starts =============================
        ERROR collecting tests/bugs/2026/test_issue_98.py
        ImportError while importing test module '/workspaces/.../tests/bugs/2026/test_issue_98.py'.
        E   ModuleNotFoundError: No module named 'some_missing_dep'
        ============================ short test summary info ============================
        ERROR tests/bugs/2026/test_issue_98.py
        !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
        ```

        Policy recommendation: treat this as **unexpected** and fail the gate.

  - Apply this filtering at both QC gates:
    - Pre-flight QC in `cli.py` (`_run_preflight_qc_with_capture` and `_run_preflight_qc_fix_loop`).
    - Phase-end QC in `QCRunner` (`run_full` and the full loop helper), so phase gates are also plan-aware.

- **Test Strategy (concrete, repo-aligned)**:
  - Parser-level tests (fast, deterministic):
    - Extend `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py` to cover:
      - `[expect-pass]` parsing
      - `test_ref` extraction
      - tag stripping from the “human title” (prompt cleanliness)

  - Expectation resolution tests (pure logic):
    - Add unit tests for a helper like `resolve_checked_test_expectations(plan: PlanModel) -> TestExpectationPolicy`.
    - Assert override behavior: when both checked `[expect-fail]` and checked `[expect-pass]` exist for same `test_ref`, the resulting expectation is **pass**.

  - Pytest output parsing tests (pure logic):
    - Add unit tests for a helper like `parse_pytest_failing_nodeids(output: str) -> set[str]`.
    - Feed representative pytest outputs (no filesystem, no real pytest invocation).

  - Gate integration tests (still unit-level via mocks):
    - In `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestPreflightQC`, patch `subprocess.run` so the pytest step returns a synthetic failing output.
    - Assert that when the failing nodeid matches a checked `[expect-fail] <test_ref>` with no checked `[expect-pass]`, preflight QC is treated as “pass” for the gating decision.
    - Assert that when a checked `[expect-pass]` exists for the same test_ref, the same failing nodeid is treated as a gate failure.

- **Dependencies**: none.

- **Success Criteria**:
  - A unit test demonstrates that a failing test listed in a checked `[expect-fail] <test_ref>` task does **not** fail the preflight or phase gate.
  - A unit test demonstrates that once a checked `[expect-pass] <test_ref>` exists for the same test_ref, any failure of that test **does** fail the gate.
  - Any failing test not listed in checked expectation tasks continues to fail the gate.

- **Notes / Clarifications**:
  - This approach intentionally uses the plan as the source of truth (not pytest markers) and therefore requires a plan convention for test nodeids.
  - The original spec’s plan-aware `--deselect` suggestion is related but not sufficient on its own; this approach also needs post-processing to decide whether a non-zero pytest exit is acceptable.

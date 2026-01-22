# 2026-01-21-capture-bats-coverage — Spec

- **Issue:** #109
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T22-20
- **Status:** Draft
- **Version:** 0.1

## Overview

We have a strong quality + tooling “ecosystem” for PowerShell (format + lint + test + Pester reporting),
and for Python (Black + Ruff + Pyright + Pytest + coverage).

For Bash, we currently have **Shell QC** (`shell-qc format/check/test`) which validates formatting
and shellcheck quality, and can run Bats tests when present.

However, we do not currently have an equivalent, standardized way to:

- Capture and persist **coverage metrics** for Bash script execution (especially Bats-driven tests).
- Generate **human-readable HTML** and machine-readable coverage summaries for CI artifacts.
- Enforce consistent defaults (paths included/excluded, report locations, and “fail on low coverage”).

This makes it hard to answer basic questions like:

- “Are our shell scripts actually exercised by tests?”
- “Which scripts / lines are not covered?”
- “Did a change decrease Bash coverage?”

The result is an uneven developer experience: Python/PowerShell have clear coverage workflows,
but Bash does not.


## Behavior

Add a first-class Bash coverage workflow that mirrors the PowerShell/Python “ecosystem”:

1. **Single entry point** to run Bats tests with coverage instrumentation.
2. Writes deterministic artifacts under `artifacts/` (and optionally `htmlcov/` or a dedicated folder).
3. Produces:
	- An **HTML report** for local inspection.
	- A **summary (text/JSON)** suitable for CI logs and future gating.
4. Integrates with existing tooling:
	- Callable via `poetry run python -m scripts.dev_tools...` (preferred) and/or a thin wrapper script.
	- Optionally wired into `Fix All` / `Run All Checks` as an opt-in phase (or default if stable).
5. Uses a **well-understood coverage mechanism** for shell scripts (e.g., `kcov` or another
	industry-standard tool) with clear include/exclude rules to avoid “coverage noise”.

Non-goals (at least initially):

- Achieving 100% Bash coverage across the repo immediately.
- Rewriting existing shell scripts to be “more testable” beyond small, targeted seams.
- Adding heavyweight runtime dependencies that are difficult to support in the devcontainer/CI.


## Inputs / Outputs

- Inputs (CLI flags, files, env vars)
	- CLI entry point (existing): `poetry run shell-qc`
	- Subcommand (existing): `test`
	- New flag (proposed): `--coverage`
		- When present, `shell-qc test` runs Bats under a coverage instrumentation tool.
		- When absent, behavior remains unchanged (run Bats normally when `tests/shell` or `tests/bash` exist).
	- Test discovery roots (existing): `tests/shell` and `tests/bash` (repo-root relative).
	- Tooling prerequisites:
		- `bats` must be installed and on PATH.
		- A coverage tool must be installed and on PATH (recommended: `kcov`).

- Outputs (artifacts, logs, telemetry)
	- Deterministic output root: `artifacts/pester/` (aligned to existing Coverage Gutters base dir).
	- Coverage output directory (proposed default): `artifacts/pester/bash/`
		- HTML report output (kcov default behavior; directory tree under the output dir).
		- Cobertura XML: `artifacts/pester/bash/cov.xml` (via kcov `--cobertura-only`, per upstream VS Code guidance).
		- Summary output:
			- Minimum requirement: total line coverage percentage emitted to stdout for CI logs.
			- Optional: also persist a summary artifact under `artifacts/pester/bash/` (format to be finalized during implementation).
	- Error output:
		- If `--coverage` is requested and prerequisites are missing (e.g., `kcov` not installed), the command should fail fast with a clear, actionable error.

- Config keys and defaults:
	- Default include/exclude behavior is required to avoid “coverage noise”. Proposed defaults (from Issue #109 acceptance criteria):
		- Exclude: `artifacts/`, `htmlcov/`, `data/`, `.venv/`, `node_modules/`.
		- Prefer include rules that scope to repo-owned shell entry points (e.g., `scripts/**/*.sh`, `scripts/**/*.bash`).
	- Coverage Gutters integration:
		- The repo already includes `cov.xml` in `coverage-gutters.coverageFileNames` and searches under `artifacts/pester/**`.
		- This feature should generate the Cobertura output in a location discoverable via that existing search path.

- Versioning or backward-compatibility constraints:
	- Must not break existing `shell-qc test` behavior when `--coverage` is not used.
	- Coverage gating (fail on low coverage) is explicitly out-of-scope for the first iteration (reporting-only rollout).

## API / CLI Surface

List commands, flags, request/response shapes, and examples.

- Command: `shell-qc test`
	- Behavior (existing):
		- If `tests/shell` or `tests/bash` exists, run Bats tests.
		- Otherwise, skip tests.

- Command: `shell-qc test --coverage` (proposed)
	- Behavior:
		- Runs Bats tests under coverage instrumentation (recommended: `kcov`).
		- Writes coverage artifacts under `artifacts/pester/` in a deterministic location.
		- Emits a machine-readable coverage file (`cov.xml`) for editor tooling (Coverage Gutters).
		- Emits a human-readable HTML report for local inspection.

- Example invocations with expected outputs (concise):
	- Local run (reporting only):
		- `poetry run shell-qc test --coverage`
		- Expected:
			- Non-zero coverage results when representative Bats tests exist.
			- `artifacts/pester/bash/cov.xml` exists for Coverage Gutters consumption.
			- HTML report is present under `artifacts/pester/bash/`.
	- CI run:
		- Same command; summary appears in CI logs and artifacts are uploaded.

- Contracts and validation rules:
	- If `--coverage` is provided:
		- Missing `bats` should be treated as an error (tests cannot run).
		- Missing coverage tool (e.g., `kcov`) should be treated as an error with guidance.
	- If `--coverage` is not provided:
		- Preserve existing “skip if no tests” behavior.

## Data & State

Data flow, storage, or state changes introduced by this feature.

- Primary state change: generating coverage artifacts under `artifacts/`.
	- This is tooling output only; there is no new persisted application state.
	- Output must be deterministic and should overwrite prior outputs in the same directory.

- Data transformations and invariants:
	- Coverage instrumentation runs the same Bats tests, but under tracing.
	- Invariant: `shell-qc test --coverage` must not modify repo source files.

- Caching or persistence details:
	- None (beyond writing artifacts under `artifacts/`).

- Migration or backfill requirements (if any):
	- None.

## Constraints & Risks

- **Tool choice / portability:**
	- Shell coverage is less standardized than Python/PowerShell. Tools like `kcov` may not be
		available everywhere by default.
	- We likely want devcontainer + CI parity first, with best-effort support for local non-container
		setups.

- **Performance / runtime cost:**
	- Coverage instrumentation can slow shell tests and inflate CI time.
	- We should support a fast “no-coverage” path (existing `shell-qc test`) and a separate
		“coverage” path.

- **Correctness of coverage signal:**
	- Bash coverage can over/under-report depending on how scripts are executed (sourced vs executed),
		subshells, and `set -euo pipefail` behavior.
	- We’ll need clear rules for “what counts as covered” and avoid misleading results.

- **Repo policy constraints:**
	- Unit tests are not allowed to create temporary files. Coverage tools typically write output
		files/directories; we should treat Bash coverage as an integration/tooling workflow, not a unit
		test constraint violation (still: outputs should go to `artifacts/` deterministically).

- **Scoping/gating risk:**
	- If we introduce coverage thresholds too early, we may block unrelated changes.
	- Start with reporting only; consider gating later once baseline coverage stabilizes.


## Implementation Strategy

- Implementation scope (what changes, not sequencing):
	- Extend Shell QC to support a coverage-enabled Bats run.
	- Ensure coverage artifacts are produced in a location compatible with existing Coverage Gutters configuration.
	- Ensure devcontainer (and CI) can run the coverage tool with minimal friction.
	- Add a minimal, representative set of Bats tests under the expected discovery directory so the workflow is exercised end-to-end.
	- Update developer documentation for the new workflow.

- New classes/functions/commands to add or update:
	- `scripts/dev_tools/shell_qc.py`
		- Add argument parsing for a coverage mode (proposed: `shell-qc test --coverage`).
		- Add a code path that executes Bats under coverage instrumentation (recommended: kcov `--cobertura-only`).
		- Preserve existing `check`, `format`, `test` behavior.
	- `docs/developer-tooling.md`
		- Document prerequisites, commands, and output locations for Bash coverage.
	- Tests and fixtures:
		- Add a repo-root Bats test directory (`tests/shell` or `tests/bash`) so Shell QC does not skip by default.
		- Update `tests/scripts/dev_tools/test_shell_qc.py` to cover the new argument parsing and error cases (e.g., `--coverage` requested but coverage tool missing).

- Dependency changes (new/removed packages) and rationale:
	- Add/install `kcov` in the devcontainer (recommended).
		- Rationale: kcov is documented upstream as producing `cov.xml` via `--cobertura-only` and supports Bash coverage.
		- Constraint: `kcov` is not available via `apt` in the current devcontainer base; installation must use a non-apt route (multi-stage copy from `kcov/kcov` image, or build-from-source during image build).

- Logging/telemetry additions and locations:
	- Emit a concise coverage summary to stdout for CI logs.
	- On failures (missing tools, non-zero test exit), print actionable errors including which tool is missing and what to install.

- Rollout plan (feature flags, staged deploys, fallback path):
	- Stage 1 (this feature): reporting-only coverage workflow (`shell-qc test --coverage`) producing artifacts and log summaries.
	- Stage 2 (future): optional gating with thresholds (explicitly non-goal for initial release).
	- Fallback: developers and CI can continue using `shell-qc test` without coverage.

## Definition of Done

- [ ] Acceptance criteria documented and mapped to tests or demos
- [ ] Behavior matches acceptance criteria in all documented environments
- [ ] Tests updated/added (unit/integration as applicable)
- [ ] Edge cases and error handling covered by tests
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Telemetry/logging added or updated (if applicable)
- [ ] Toolchain pass completed (format → lint → type-check → test)

## Seeded Test Conditions (from potential)
- [ ] Coverage includes a script invoked directly (e.g., `scripts/.../tool.sh`).
- [ ] Coverage includes a sourced helper library (e.g., `scripts/.../lib.sh`) when used by tests.
- [ ] A script that intentionally exits non-zero is reported correctly and doesn’t corrupt the report.
- [ ] Paths excluded from coverage (e.g., `artifacts/`, `.venv/`) do not appear in reports.
- [ ] Running twice produces stable output locations (old reports are overwritten or cleaned).
- [ ] Works in devcontainer and in CI runner environment.
- [ ] Optional: demonstrate a baseline report on a small set of Bats tests so we can confirm
- [ ] coverage is captured end-to-end.

# capture-bats-coverage (Issue #109)

- Date captured: 2026-01-21
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/capture-bats-coverage/ (Issue #109)

- Issue: #109
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/109
- Last Updated: 2026-01-22
## Problem / Why

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

## Proposed Behavior

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

## Acceptance Criteria (early draft)

- [ ] A new developer-facing command exists to run Bats tests **with coverage enabled** (one command,
	documented), producing consistent output paths under `artifacts/`.
- [ ] The coverage run generates both:
	- an **HTML report** (for browsing locally), and
	- a **machine-readable summary** (JSON or similar) that includes at least total line coverage %.
- [ ] Coverage instrumentation supports common repo script patterns:
	- scripts executed directly
	- scripts sourced by other scripts
	- functions invoked from Bats tests
- [ ] Coverage output excludes noise by default (documented):
	- excludes `artifacts/`, `htmlcov/`, `data/`, `.venv/`, `node_modules/`, etc.
	- limits inclusion to repo-owned Bash entry points (e.g., `scripts/**/*.sh`, `scripts/**/*.bash`)
- [ ] CI can run the Bash coverage workflow without interactive prompts and uploads the report as an
	artifact (or publishes summary to logs), without breaking existing quality gates.
- [ ] A minimal set of representative Bats tests exist (or are extended) such that coverage is
	demonstrably non-zero and the reporting pipeline is validated.
- [ ] Documentation exists in `docs/developer-tooling.md` (or a new short doc) describing:
	- prerequisites/tools
	- how to run locally
	- where reports are written
	- how to interpret results

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

## Test Conditions to Consider

- [ ] Coverage includes a script invoked directly (e.g., `scripts/.../tool.sh`).
- [ ] Coverage includes a sourced helper library (e.g., `scripts/.../lib.sh`) when used by tests.
- [ ] A script that intentionally exits non-zero is reported correctly and doesn’t corrupt the report.
- [ ] Paths excluded from coverage (e.g., `artifacts/`, `.venv/`) do not appear in reports.
- [ ] Running twice produces stable output locations (old reports are overwritten or cleaned).
- [ ] Works in devcontainer and in CI runner environment.
- [ ] Optional: demonstrate a baseline report on a small set of Bats tests so we can confirm
	coverage is captured end-to-end.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
	- Include tool recommendation (e.g., `kcov`) and rollout plan (reporting-only → optional gating).
- [ ] Create `docs/features/active/capture-bats-coverage/` folder from the template
	- Draft `spec.md` describing the command(s), output paths, and CI integration points.

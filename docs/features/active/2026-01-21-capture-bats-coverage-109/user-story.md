# `2026-01-21-capture-bats-coverage` — User Story

- Issue: #109
- Owner: drmoisan
- Status: Draft
- Last Updated: 2026-01-21T22-20

## Story Statement

- As a repo contributor writing or maintaining Bash scripts, I want a single command to run Bats tests with coverage and produce deterministic reports, so that I can see which shell code paths are exercised and iterate quickly.
- As a CI maintainer, I want Bash coverage to emit machine-readable outputs and a concise summary without interactive prompts, so that CI can archive reports and we can later introduce non-breaking coverage baselines and gating.

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


## Personas & Scenarios

- Persona: Repo contributor (Bash script author)
  - Who: A developer who primarily works in Python/PowerShell in this repo but occasionally edits `scripts/**/*.sh`.
  - Cares about: Consistent “ecosystem” workflows (one command; deterministic artifacts under `artifacts/`), fast feedback, and coverage visibility in VS Code.
  - Constraints: May not have local tooling installed outside the devcontainer; wants a workflow that works in the devcontainer and CI first.
  - Goals / frustrations: Today they can run `shell-qc test`, but they cannot answer “what is covered?” or see a coverage view in the editor.
  - Context / motivations: Wants parity with Python/PowerShell quality gates and to prevent regressions in shell tooling.

- Scenario: Validate coverage after changing a shell script
  - Who is acting: Repo contributor (Bash script author).
  - Trigger: They refactor a shell script used by developer tooling and want to confirm tests actually execute the changed code.
  - Steps:
    1) Run `poetry run shell-qc test --coverage`.
    2) Open the HTML report under `artifacts/` to inspect uncovered regions.
    3) Use VS Code Coverage Gutters (via `cov.xml`) to see coverage in the editor.
  - Obstacles / decisions:
    - If the coverage tool is missing (e.g., `kcov` not installed), the command should fail fast with a clear message.
    - If there are no Bats tests discovered (no `tests/shell` or `tests/bash`), the contributor needs at least one representative test added so coverage is non-zero.
  - Expected outcome: Coverage artifacts are created deterministically, coverage is visibly non-zero for the representative tests, and the contributor can identify which shell lines need test coverage.


## Acceptance Criteria

- [ ] A single developer-facing command exists to run Bats tests with coverage enabled (proposed: `poetry run shell-qc test --coverage`).
- [ ] The command produces deterministic outputs under `artifacts/` and is compatible with the repo’s existing Coverage Gutters search path (currently under `artifacts/pester/**`).
- [ ] The coverage run produces an HTML report under `artifacts/` for local inspection.
- [ ] The coverage run produces a Cobertura XML file named `cov.xml` under `artifacts/` (discoverable by Coverage Gutters).
- [ ] The coverage run emits a concise coverage summary to stdout that includes total line coverage percentage (suitable for CI logs).
- [ ] When `--coverage` is requested and the required coverage tool is missing (recommended: `kcov`), the command fails fast with an actionable error message.
- [ ] Coverage defaults exclude known “noise” directories (at minimum: `artifacts/`, `htmlcov/`, `data/`, `.venv/`, `node_modules/`) and scope inclusion to repo-owned shell entry points (e.g., `scripts/**/*.sh`, `scripts/**/*.bash`).
- [ ] A minimal set of representative Bats tests exist in a Shell QC-discoverable location (`tests/shell` or `tests/bash`) such that coverage is demonstrably non-zero.
- [ ] CI can run the coverage command without interactive prompts and upload the coverage report directory as an artifact (or otherwise retain it), without breaking existing quality gates.
- [ ] Documentation exists in `docs/developer-tooling.md` describing prerequisites, how to run locally, where reports are written, and how to interpret results.


## Non-Goals

- Enforcing a minimum coverage threshold (coverage gating) in CI as part of the initial rollout.
- Achieving 100% Bash coverage across the repo.
- Large-scale refactors of existing shell scripts solely to improve testability.
- Supporting non-devcontainer local environments as a first-class requirement before devcontainer + CI parity is stable.
- Replacing the existing `shell-qc test` workflow; the non-coverage path remains available.

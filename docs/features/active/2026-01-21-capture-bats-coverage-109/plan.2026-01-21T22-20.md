---
issue: 109
parent: none
owner: drmoisan
last_updated: 2026-01-21T22-20
status: Planned
status_color: blue
version: 0.2
---

# 2026-01-21-capture-bats-coverage - Plan

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- **Issue:** #109
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T22-20
- **Status:** Planned
- **Version:** 0.2

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Code Change Policy: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)
- GitHub Actions Workflow Policy (required if updating CI): [`.github/instructions/github-actions.instructions.md`](../../../../.github/instructions/github-actions.instructions.md)

Additional context for this plan (must be read before execution):

- Feature spec: [`docs/features/active/2026-01-21-capture-bats-coverage-109/spec.md`](./spec.md)
- User story: [`docs/features/active/2026-01-21-capture-bats-coverage-109/user-story.md`](./user-story.md)
- Research: [`docs/features/active/2026-01-21-capture-bats-coverage-109/research.md`](./research.md)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

> **Instructions for this section:**
> - Break work into **Phases** (broad buckets) and **Atomic Tasks** (binary, 5-30 min units).
> - Use `- [ ] [P#-T#]` for every task.
> - Start every task with a **strong verb** (Implement, Create, Update, Verify).
> - No "bucket" tasks like "Refactor module" or "Write tests"; split them into specific, verifiable steps.
> - **Self-Validating Phases:** Include necessary test creation/update tasks *within* the phase that implements the code. Do not defer verification to a final "Testing" phase.

Requirements Traceability (REQ → tasks)

| REQ ID | Requirement | Implemented by tasks |
| --- | --- | --- |
| REQ-001 | Add a single command to run Bats tests with coverage enabled (`poetry run shell-qc test --coverage`). | P4-T1, P4-T2, P4-T3 |
| REQ-002 | Generate deterministic coverage artifacts under `artifacts/`, including HTML output. | P4-T3 |
| REQ-003 | Emit Cobertura XML `cov.xml` discoverable by VS Code Coverage Gutters. | P4-T3 |
| REQ-004 | Emit a concise stdout summary including total line coverage percentage. | P4-T4 |
| REQ-005 | Fail fast with actionable error if coverage prerequisites are missing (kcov). | P2-T2, P4-T2 |
| REQ-006 | Default include/exclude rules avoid noise (exclude artifacts/htmlcov/data/.venv/node_modules and scope to repo-owned scripts). | P4-T3 |
| REQ-007 | Add minimal representative Bats tests under a Shell QC-discoverable root (`tests/shell` or `tests/bash`) producing non-zero coverage. | P3-T1, P3-T2, P3-T3 |
| REQ-008 | Devcontainer supports the workflow (kcov install via non-apt route). | P1-T1, P1-T2 |
| REQ-009 | CI can run the coverage command and upload the coverage artifacts as an artifact. | P5-T1, P5-T2 |
| REQ-010 | Document prerequisites and usage in `docs/developer-tooling.md`. | P6-T1 |

### Phase 0 — Context & Baseline

- [x] [P0-T1] Read `.github/copilot-instructions.md`
  - Acceptance: `test -s .github/copilot-instructions.md` exits with code 0.

- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md`
  - Acceptance: `test -s .github/instructions/general-code-change.instructions.md` exits with code 0.

- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md`
  - Acceptance: `test -s .github/instructions/general-unit-test.instructions.md` exits with code 0.

- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md`
  - Acceptance: `test -s .github/instructions/python-code-change.instructions.md` exits with code 0.

- [x] [P0-T5] Read `.github/instructions/python-unit-test.instructions.md`
  - Acceptance: `test -s .github/instructions/python-unit-test.instructions.md` exits with code 0.

- [x] [P0-T6] Read `.github/instructions/github-actions.instructions.md`
  - Acceptance: `test -s .github/instructions/github-actions.instructions.md` exits with code 0.

- [x] [P0-T7] Create `artifacts/capture-bats-coverage-109_policy_ack.txt` documenting policy files reviewed
  - Preconditions: [P0-T1]–[P0-T6] completed.
  - Implementation details:
    - The file must contain the exact relative paths of:
      - `.github/copilot-instructions.md`
      - `.github/instructions/general-code-change.instructions.md`
      - `.github/instructions/general-unit-test.instructions.md`
      - `.github/instructions/python-code-change.instructions.md`
      - `.github/instructions/python-unit-test.instructions.md`
      - `.github/instructions/github-actions.instructions.md`
    - The file must include an ISO-8601 timestamp line prefixed with `ReviewedAt:`.
  - Acceptance: `test -s artifacts/capture-bats-coverage-109_policy_ack.txt` exits with code 0.

- [x] [P0-T8] Capture baseline Ruff results to `artifacts/capture-bats-coverage-109_baseline_ruff.txt`
  - Acceptance: `poetry run ruff check | tee artifacts/capture-bats-coverage-109_baseline_ruff.txt` exits with code 0.

- [x] [P0-T9] Capture baseline Pyright results to `artifacts/capture-bats-coverage-109_baseline_pyright.txt`
  - Acceptance: `poetry run pyright | tee artifacts/capture-bats-coverage-109_baseline_pyright.txt` exits with code 0.

- [x] [P0-T10] Capture baseline Pytest results to `artifacts/capture-bats-coverage-109_baseline_pytest.txt`
  - Acceptance: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing | tee artifacts/capture-bats-coverage-109_baseline_pytest.txt` exits with code 0.

- [x] [P0-T11] Capture baseline actionlint results to `artifacts/capture-bats-coverage-109_baseline_actionlint.txt`
  - Acceptance: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-actionlint.ps1 | tee artifacts/capture-bats-coverage-109_baseline_actionlint.txt` exits with code 0.

### Phase 1 — Devcontainer support (kcov install)

- [x] [P1-T1] Update `.devcontainer/Dockerfile` to install `kcov` via multi-stage copy from `kcov/kcov:latest` (non-apt route)
  - Implementation details:
    - Add a build stage: `FROM kcov/kcov:latest AS kcov`
    - Add a copy step: `COPY --from=kcov /usr/local/bin/kcov* /usr/local/bin/`
    - Keep the base stage as `mcr.microsoft.com/devcontainers/python:3.13-bookworm`.
  - Acceptance: After rebuilding the devcontainer, `kcov --version` succeeds inside the container.

- [x] [P1-T2] Update `.devcontainer/Dockerfile` to ensure `bats` is installed during devcontainer build
  - Acceptance: After rebuilding the devcontainer, `bats --version` succeeds inside the container.

### Phase 2 — TDD red tests (Python unit tests)

- [x] [P2-T1] [expect-fail] Add a failing pytest test asserting `scripts.dev_tools.shell_qc.parse_args(["test", "--coverage"])` is accepted in `tests/scripts/dev_tools/test_shell_qc.py`
  - Preconditions: No production code changes in `scripts/dev_tools/shell_qc.py` have been made for `--coverage`.
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_shell_qc.py -k test_parse_args_accepts_test_coverage_flag` fails.

- [x] [P2-T2] [expect-fail] Add a failing pytest test asserting `scripts.dev_tools.shell_qc.main(["test", "--coverage"])` returns exit code 127 when `kcov` is missing in `tests/scripts/dev_tools/test_shell_qc.py`
  - Preconditions: No production code changes have been made to enforce `kcov` presence.
  - Implementation details:
    - Use `monkeypatch.setattr(shell_qc.shutil, "which", lambda _: None)` to simulate missing `kcov` deterministically.
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_shell_qc.py -k test_main_test_coverage_errors_when_kcov_missing` fails.

### Phase 3 — Add minimal Bats tests + coverage demo scripts

- [ ] [P3-T1] Add a minimal shell library under `scripts/bash/coverage_lib.sh` with at least two functions (one called by tests, one intentionally unused)
  - Acceptance: File exists at `scripts/bash/coverage_lib.sh` and passes Shell QC discovery (will be included by `discover_shell_scripts`).

- [ ] [P3-T2] Add a minimal shell entry point under `scripts/bash/coverage_demo.sh` that sources `coverage_lib.sh` and invokes the covered function
  - Acceptance: File exists at `scripts/bash/coverage_demo.sh` and runs successfully under `bash scripts/bash/coverage_demo.sh`.

- [ ] [P3-T3] Add a Bats test under `tests/shell/test_coverage_demo.bats` that executes `scripts/bash/coverage_demo.sh`
  - Acceptance: `poetry run shell-qc test` exits with code 0.

### Phase 4 — Implement `shell-qc test --coverage`

- [x] [P4-T1] Update `scripts/dev_tools/shell_qc.py:parse_args` to add a `--coverage` flag for the `test` subcommand
  - Implementation details:
    - Capture the `test` parser instance returned by `subparsers.add_parser("test", ...)`.
    - Add `test_parser.add_argument("--coverage", action="store_true", help="Run bats tests under coverage instrumentation.")`.
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_shell_qc.py -k test_parse_args_accepts_test_coverage_flag` exits with code 0.

- [x] [P4-T2] Update `scripts/dev_tools/shell_qc.py:main` to dispatch `test --coverage` to a new coverage-aware path
  - Implementation details:
    - If `args.command == "test"` and `args.coverage is True`, call a new function `run_test_with_coverage()`.
    - If `args.command == "test"` and `args.coverage is False`, preserve existing `run_test()` behavior.
    - In `run_test_with_coverage()`, require `kcov` via `shutil.which("kcov")` and return 127 if missing.
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_shell_qc.py -k test_main_test_coverage_errors_when_kcov_missing` exits with code 0.

- [x] [P4-T3] Implement `scripts/dev_tools/shell_qc.py:run_test_with_options` to generate `artifacts/pester/kcov/cov.xml` and HTML output using `kcov --cobertura-only`
  - Implementation details:
    - Coverage output directory: `Path("artifacts") / "pester" / "bash"`.
    - Ensure the output directory exists before invoking kcov.
    - Use `find_bats_test_dirs()` for test discovery (same as `run_test`).
    - For each discovered test directory, execute kcov wrapping bats:
      - Command shape: `kcov --cobertura-only --include-path=<repo>/scripts --include-path=<repo>/tools --exclude-pattern=artifacts,htmlcov,data,.venv,node_modules,.git,dist,build <out_dir> bats <test_dir>`
    - If more than one test directory is present, run each into a deterministic per-dir output (e.g., `<out_dir>/run_0`, `<out_dir>/run_1`) and then merge into `<out_dir>` using `kcov --merge`.
  - Acceptance: `poetry run shell-qc test --coverage` exits with code 0 in the devcontainer and writes `artifacts/pester/bash/cov.xml`.

- [ ] [P4-T4] Add a deterministic stdout summary emitted after coverage generation by parsing the generated `cov.xml` line-rate
  - Implementation details:
    - Add a helper function `extract_cobertura_line_rate(cov_xml: str) -> float` that reads `cov.xml` as text and extracts the first `line-rate="..."` attribute.
    - Print a single-line summary: `Bash coverage (lines): <percent>%`.
  - Acceptance: After `poetry run shell-qc test --coverage`, stdout contains `Bash coverage (lines):`.

### Phase 5 — CI integration (artifact upload)

- [ ] [P5-T1] Update `.github/workflows/ci.yml` to add a new Linux job `shell-coverage` that runs `poetry run shell-qc test --coverage`
  - Implementation details:
    - Install required OS tools (at minimum): `shellcheck`, `shfmt`, `bats`.
    - Install `kcov` in the job using the upstream build-from-source dependencies from `docs/features/active/2026-01-21-capture-bats-coverage-109/research.md`.
    - Run `poetry install --no-interaction` before invoking `shell-qc`.
  - Acceptance: The workflow contains a `shell-coverage` job and the job includes a step that runs `poetry run shell-qc test --coverage`.

- [ ] [P5-T2] Update `.github/workflows/ci.yml` to upload `artifacts/pester/bash/**` using `actions/upload-artifact@v4`
  - Acceptance: The `shell-coverage` job includes an upload step with `path: artifacts/pester/bash/**` and `if-no-files-found: error`.

### Phase 6 — Documentation

- [ ] [P6-T1] Update `docs/developer-tooling.md` to document the Bash coverage workflow
  - Implementation details:
    - Document the command: `poetry run shell-qc test --coverage`.
    - Document output locations: `artifacts/pester/bash/` and `artifacts/pester/bash/cov.xml`.
    - Document prerequisites: `bats` and `kcov` (and note that kcov is installed via devcontainer build, not apt).
  - Acceptance: `docs/developer-tooling.md` contains a section that mentions `shell-qc test --coverage` and `artifacts/pester/bash/cov.xml`.

### Phase 7 — Final QA (toolchain loop)

- [ ] [P7-T1] Run formatting and confirm no changes are required
  - Acceptance: `poetry run black .` exits with code 0 and does not modify files.

- [ ] [P7-T2] Run Ruff linting and confirm no errors
  - Acceptance: `poetry run ruff check` exits with code 0.

- [ ] [P7-T3] Run Pyright strict type checking and confirm no errors
  - Acceptance: `poetry run pyright` exits with code 0.

- [ ] [P7-T4] Run Pytest (with coverage) and confirm all tests pass
  - Acceptance: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` exits with code 0.

- [ ] [P7-T5] Re-run `poetry run shell-qc test --coverage` and confirm artifacts are generated
  - Acceptance: Command exits with code 0 and `artifacts/pester/bash/cov.xml` exists.

- [ ] [P7-T6] Run actionlint and confirm workflow changes remain valid
  - Acceptance: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-actionlint.ps1` exits with code 0.

## Test Plan

- Unit (pytest; deterministic; no filesystem temp files):
  - `tests/scripts/dev_tools/test_shell_qc.py::test_parse_args_accepts_test_coverage_flag`
  - `tests/scripts/dev_tools/test_shell_qc.py::test_main_test_coverage_errors_when_kcov_missing`

- Integration (tooling workflow; writes deterministic outputs under `artifacts/`):
  - `poetry run shell-qc test` (executes Bats tests from `tests/shell`)
  - `poetry run shell-qc test --coverage` (generates `artifacts/pester/bash/cov.xml`)

- Manual/CLI (non-gating convenience checks; do not block completion):
  - Open `artifacts/pester/bash/index.html` (or equivalent kcov HTML entrypoint) in a browser to inspect uncovered lines.

## Open Questions / Notes

- kcov install guidance:
  - `kcov` is not available via `apt-get install kcov` in the current devcontainer base; this plan installs it via Docker multi-stage copy.

- Coverage Gutters configuration:
  - The repo already includes `cov.xml` in Coverage Gutters recognized filenames and uses `artifacts/pester/**` as the base dir.
  - This plan writes Bash coverage to `artifacts/pester/bash/cov.xml` to avoid editor configuration changes.

- CI feasibility:
  - If kcov build-from-source in GitHub Actions proves too slow, revise Phase 5 to use a prebuilt kcov binary source, but keep job behavior and artifact paths unchanged.

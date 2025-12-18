# fix-all-script (Plan)

- Issue: #44
- Owner: Dan Moisan
- Date: 2025-12-15
- Status: Draft

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Link approved spec: docs/features/active/fix-all-script/spec.md
- [x] [P0-T2] Record branch/commit baseline for development work (branch: work, commit: 9f5cbd9795f017d92e7e6de8f6c672119aed42ce)
- [x] [P0-T3] Note required environment: repo dev container (Poetry, Black, Ruff, Pyright, Pytest available)

**Phase 1 — Design & Parity Decisions**
- [x] [P1-T1] Decide Python module location/entrypoint for fix-all (scripts/dev_tools/fix_all.py with poetry run python -m scripts.dev_tools.fix_all)
- [x] [P1-T2] Define CLI surface and flags (retain MaxRuffRetries equivalent, verbosity) to mirror current PowerShell behavior (CLI supports --max-ruff-retries and --no-coverage)
- [x] [P1-T3] Specify command execution semantics: capture stdout/stderr, treat Black exit code 0 as success even with stderr noise, fail on non-zero exit codes
- [x] [P1-T4] Map VS Code tasks and any wrappers that must point to the Python entrypoint; plan deprecation path for fix-all.ps1 (Python is source of truth; PowerShell delegates)

**Phase 2 — Scaffolding & Helpers (Python)**
- [x] [P2-T1] Create Python module skeleton with main() and helper to run commands with status (stdout/stderr capture, exit code return)
- [x] [P2-T2] Add structured step logging equivalents (step/success/failure) in Python
- [x] [P2-T3] Implement argument parsing for MaxRuffRetries and future-proof flag for coverage toggle (default on)

**Phase 3 — Implement Toolchain Logic in Python**
- [x] [P3-T1] Implement Black step: invoke Black, capture exit code, allow stderr when exit code is 0, fail otherwise
- [x] [P3-T2] Implement Ruff auto-fix with retry loop honoring MaxRuffRetries; surface failures with attempt count
- [x] [P3-T3] Implement Black verify pass and Ruff verify pass mirroring current sequencing
- [x] [P3-T4] Implement Pyright step with fail-fast on non-zero exit
- [x] [P3-T5] Implement Pytest step with coverage flags `--cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
- [x] [P3-T6] Aggregate exit reporting to mirror current “ALL CHECKS PASSED” summary

**Phase 4 — Testing Plan (Pytest scenarios)**
- [x] [P4-T1] Add unit test for command runner returning 0 when exit code 0 and stderr present (bug repro guard) in tests/dev_tools/test_fix_all.py
- [x] [P4-T2] Add unit test for command runner failing on non-zero exit with stderr surfacing
- [x] [P4-T3] Add unit test for Black step success path (mock command runner exit 0 with stderr) to ensure no failure
- [x] [P4-T4] Add unit test for Black step failure (exit != 0) to ensure failure propagates
- [x] [P4-T5] Add unit test for Ruff retry logic: first attempt fails, second succeeds, stops further retries
- [x] [P4-T6] Add unit test for Ruff exhausting retries and returning failure
- [x] [P4-T7] Add unit test for Pyright failure handling (exit != 0 stops pipeline)
- [x] [P4-T8] Add unit test for Pytest failure handling (exit != 0 stops pipeline)
- [x] [P4-T9] Add integration-style unit test wiring all mocked steps to succeed to ensure summary and exit code 0
- [x] [P4-T10] Ensure coverage ≥ 91% for new Python module (measure via pytest --cov) and remove dependence on PowerShell coverage

**Phase 5 — Migration & Parity Checks**
- [x] [P5-T1] Update VS Code task “Dev: 0 Fix All” (and any wrappers) to call the Python entrypoint
- [x] [P5-T2] Add compatibility shim in fix-all.ps1 to delegate to Python or emit deprecation notice (while preserving current callers)
- [x] [P5-T3] Remove or gate POSHQC_SKIP_SCRIPT_EXECUTION logic if no longer needed for Python path (document behavior)
- [x] [P5-T4] Validate task output text matches (or improves on) existing PowerShell messaging to reduce confusion

**Phase 6 — Verification Loop**
- [x] [P6-T1] Run Black → Ruff → Pyright → Pytest on the repo using the new Python fix-all to confirm green pipeline
- [x] [P6-T2] Confirm regression: a normal Black success with stderr does not fail the run (original bug condition)
- [x] [P6-T3] Capture coverage report showing ≥91% for the new Python module

**Phase 7 — Documentation & Rollout**
- [x] [P7-T1] Update README/docs/developer-tooling to reference the Python fix-all entrypoint and deprecate the PowerShell version
- [x] [P7-T2] Note behavior changes in issue #44 and spec (stderr handling, retry logic parity)
- [x] [P7-T3] Remove or archive redundant PowerShell-specific tests if replaced; ensure new pytest tests are listed in coverage artifacts
- [ ] [P7-T4] Prepare PR notes (summary, risks, validation, links to tests) and request review

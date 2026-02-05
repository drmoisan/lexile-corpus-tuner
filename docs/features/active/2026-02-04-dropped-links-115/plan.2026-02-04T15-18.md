---
title: "2026-02-04-dropped-links (Plan)"
issue: "115"
parent: "none"
owner: "drmoisan"
last_updated: "2026-02-04T18-59"
status: "Completed"
status_color: "green"
version: "1.0"
---

# 2026-02-04-dropped-links (Plan)

- **Issue:** #115
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-02-04T18-59
- **Status:** Completed
- **Version:** 1.0

Status Badge: Completed [green]

Purpose: ensure CK-12 URL validation succeeds by adding a browser-like User-Agent to the HEAD request in `validate_url()` while preserving existing CK-12 selection and content-type validation rules.

**Requirements Traceability**

| ID | Requirement | Source |
| --- | --- | --- |
| REQ-001 | Add a browser-like `User-Agent` header to the HEAD request in `validate_url()` so CK-12 revision URLs validate. | spec.md: Proposed Fix + Root Cause Analysis |
| REQ-002 | Preserve HEAD-based validation flow and existing content-type prefix checks for all sources. | spec.md: Boundaries and invariants to preserve |
| REQ-003 | Do not add new dependencies; use standard library `urllib.request`. | spec.md: Dependencies or blocked work |
| REQ-004 | Add a unit test that asserts the `User-Agent` header is set when calling `validate_url()`. | spec.md: Test Strategy |
| REQ-005 | Add unit tests for edge cases: non-200 status and content-type mismatch. | spec.md: Test Strategy + Edge cases |
| REQ-006 | Complete Python toolchain loop (Black → Ruff → Pyright → Pytest) after changes. | spec.md: Toolchain commands |
| REQ-007 | Re-run `oer_manifest --validate-urls` and capture non-zero CK-12 entries. | spec.md: Repro & Evidence + Rollout |

**Constraints & Security Notes**

| ID | Constraint / Security Note | Source |
| --- | --- | --- |
| CON-001 | Keep HEAD requests; do not switch to GET. | spec.md: Out of scope / non-goals |
| CON-002 | Preserve CK-12 candidate selection rules and filename conventions. | spec.md: Boundaries and invariants to preserve |
| SEC-001 | URL validation targets trusted CK-12 HTTPS endpoints; continue to use `urllib.request` with a fixed User-Agent header. | spec.md: Assumptions, Constraints, Dependencies |

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read `.github/copilot-instructions.md` and acknowledge in `docs/features/active/2026-02-04-dropped-links-115/baseline/policy-ack.md`.
	- Acceptance: `policy-ack.md` exists and includes the literal line `copilot-instructions.md: read`.
- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` and append acknowledgment to `baseline/policy-ack.md`.
	- Acceptance: `policy-ack.md` includes the literal line `general-code-change.instructions.md: read`.
- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` and append acknowledgment to `baseline/policy-ack.md`.
	- Acceptance: `policy-ack.md` includes the literal line `general-unit-test.instructions.md: read`.
- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` and `.github/instructions/python-unit-test.instructions.md`; append acknowledgments to `baseline/policy-ack.md`.
	- Acceptance: `policy-ack.md` includes the literal lines `python-code-change.instructions.md: read` and `python-unit-test.instructions.md: read`.
- [x] [P0-T5] Capture baseline Git context in `docs/features/active/2026-02-04-dropped-links-115/baseline/git-context.txt`.
	- Acceptance: `git-context.txt` contains the output of `git rev-parse --abbrev-ref HEAD` and `git rev-parse HEAD`.
- [x] [P0-T6] Capture baseline Python toolchain outputs to `docs/features/active/2026-02-04-dropped-links-115/baseline/`.
	- Acceptance: `baseline/black.txt`, `baseline/ruff.txt`, `baseline/pyright.txt`, and `baseline/pytest.txt` exist and each includes the exact command line used.
	- Commands: `poetry run black .`, `poetry run ruff check`, `poetry run pyright`, `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`.
- [x] [P0-T7] Record required data inputs in `docs/features/active/2026-02-04-dropped-links-115/baseline/inputs.md`.
	- Acceptance: `inputs.md` lists `data/meta/catalogs/ck12_curated.jsonl` and the CLI command `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`.
- [x] [P0-T8] Read `docs/features/active/2026-02-04-dropped-links-115/research.md` and acknowledge in `docs/features/active/2026-02-04-dropped-links-115/baseline/research-ack.md`.
	- Acceptance: `research-ack.md` exists and includes the literal line `research.md: read`.

### Phase 1 — Preparation
- [x] [P1-T1] Confirm scope lock by copying the current `spec.md` hash into `docs/features/active/2026-02-04-dropped-links-115/baseline/spec-sha256.txt`.
	- Acceptance: `spec-sha256.txt` contains the output of `sha256sum docs/features/active/2026-02-04-dropped-links-115/spec.md`.
- [x] [P1-T2] Verify target file locations and line references for the change set.
	- Acceptance: `docs/features/active/2026-02-04-dropped-links-115/baseline/targets.md` lists `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py: validate_url() (approx line 143)` and `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py (approx lines 119–166)`.
- [x] [P1-T3] Create the canonical regression evidence directory `docs/features/active/2026-02-04-dropped-links-115/regression-testing/`.
	- Acceptance: `docs/features/active/2026-02-04-dropped-links-115/regression-testing/` exists.

### Phase 2 — Regression Tests (TDD Red)
- [x] [P2-T1] [expect-fail] Add pytest test `test_validate_url_sets_user_agent_header` in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` that asserts `urllib.request.Request` receives `headers={"User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"}` when `validate_url("https://www.ck12.org/flx/get/detail/revision/123?tiny=true")` is invoked.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k user_agent_header` fails with an assertion that the header is missing or mismatched, and `docs/features/active/2026-02-04-dropped-links-115/regression-testing/pytest-user_agent_header.fail.txt` exists with fields `Timestamp:`, `Command:`, `EXIT_CODE:` (non-zero) plus a `Failure:` excerpt.
	- REQ: REQ-004
- [x] [P2-T2] Add pytest test `test_validate_url_still_uses_head_method` in the same file that asserts the Request method is `"HEAD"` for CK-12 validation URLs.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k head_method` exits with code 0 and `docs/features/active/2026-02-04-dropped-links-115/regression-testing/pytest-head_method.pass.txt` exists with fields `Timestamp:`, `Command:`, `EXIT_CODE:` (0).
	- Rationale: Baseline already enforced HEAD; test never failed.
	- REQ: REQ-002
- [x] [P2-T3] Add pytest test `test_validate_url_rejects_non_200_status` in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` that asserts `validate_url()` returns `(False, 403, "text/plain")` when the response status is 403.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k non_200_status` exits with code 0.
	- REQ: REQ-005
- [x] [P2-T4] Add pytest test `test_validate_url_rejects_content_type_mismatch` in the same file that asserts `validate_url()` returns `False` when the Content-Type is `application/json` but allowed prefixes are `text`.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k content_type_mismatch` exits with code 0.
	- REQ: REQ-005

### Phase 3 — Minimal Fix
- [x] [P3-T1] Update `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py` in `validate_url()` (approx line 143) to include the explicit header `"User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"` in the `urllib.request.Request` call while keeping `method="HEAD"` unchanged.
	- Acceptance: `oer_manifest.py` contains `headers={"User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"}` on the `urllib.request.Request` call in `validate_url()` and still includes `method="HEAD"`.
	- REQ: REQ-001, REQ-002, REQ-003
- [x] [P3-T2] Update inline comments in `validate_url()` to explain the CK-12 CloudFront User-Agent requirement without changing behavior.
	- Acceptance: `oer_manifest.py` has a comment directly above the Request creation that includes the literal phrase `CK-12 CloudFront`.
	- REQ: REQ-001

### Phase 4 — Verification Loop
- [x] [P4-T1] Re-run the two new header/method tests and capture the passing output.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k "user_agent_header or head_method"` exits with code 0 and `docs/features/active/2026-02-04-dropped-links-115/regression-testing/pytest-user_agent_header.pass.txt` exists with fields `Timestamp:`, `Command:`, `EXIT_CODE:` (0).
- [x] [P4-T2] Re-run the edge-case tests and capture the passing output.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k "non_200_status or content_type_mismatch"` exits with code 0 and `docs/features/active/2026-02-04-dropped-links-115/regression-testing/pytest-validate_url-edgecases.pass.txt` exists with fields `Timestamp:`, `Command:`, `EXIT_CODE:` (0).
- [x] [P4-T3] Execute a full Python toolchain loop in order and repeat from Black if any step modifies files or fails.
	- Acceptance: `poetry run black .`, `poetry run ruff check`, `poetry run pyright`, and `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` complete with exit code 0 in a single pass and the outputs are saved to `docs/features/active/2026-02-04-dropped-links-115/qa/black.txt`, `qa/ruff.txt`, `qa/pyright.txt`, and `qa/pytest.txt`.
	- REQ: REQ-006

### Phase 5 — Documentation & Status
- [x] [P5-T1] Update `docs/features/active/2026-02-04-dropped-links-115/spec.md` to mark acceptance criteria items that are now satisfied.
	- Acceptance: `spec.md` checkboxes for regression tests, toolchain pass, and repro steps are marked `[x]`.
- [x] [P5-T2] Update this plan file status to `In progress` or `Completed` and bump `Last Updated`.
	- Acceptance: `plan.2026-02-04T15-18.md` has `Status:` set to the correct value and `Last Updated:` updated to the current timestamp.

### Phase 6 — PR & Handoff
- [x] [P6-T1] Record PR summary notes in `docs/features/active/2026-02-04-dropped-links-115/pr-notes.md`.
	- Acceptance: `pr-notes.md` contains sections titled `Summary`, `Risk`, `Validation`, and `Related Links`.

### Phase 7 — Rollout / Follow-up
- [x] [P7-T1] Capture post-fix verification output for the CLI repro in `docs/features/active/2026-02-04-dropped-links-115/regression-testing/ck12-manifest.pass.txt`.
	- Acceptance: `ck12-manifest.pass.txt` includes the exact command line, `Timestamp:`, `Command:`, `EXIT_CODE:` (0), and a line starting with `Manifest entries:` followed by a non-zero integer.
	- REQ: REQ-007

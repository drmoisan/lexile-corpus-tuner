# Code Review — populate-open-stax-ck-12-manifest-73 (v4)

## Summary

The branch delivers a substantial OER pipeline for OpenStax/CK-12 (catalog → enrich → curate → manifest → download → extract). Python linting/type-checking/tests pass; PowerShell linting/tests pass; JSON validation passes. Bash tests fail (3 failures), new-module coverage is below policy thresholds, and multiple files exceed the 500-line limit. **Recommendation: No-go** until remediation.

Top risks:
- Bash CI failures (`shell-qc test`) block merge readiness.
- New-module coverage below 90% violates policy and reduces confidence in OER/CK-12 behaviors.
- Oversized modules (>500 lines) violate repo policy and reduce maintainability.

## Findings

| Severity | File | Location | Finding | Recommendation | Rationale | Evidence |
|---|---|---|---|---|---|---|
| Blocker | `tests/shell/test_agent_mvp.bats` | tests 2–4 | Bash test failures in `shell-qc test` | Fix failing shell tests before merge | CI gate fails; policy requires clean toolchain pass | `shell-qc test` output shows 3 failures (status 3/4/5 expectations) |
| Major | `ck12_enrichment.py`, `ck12_catalog.py`, `oer_catalog.py`, `oer_enrichment.py`, `oer_curation.py`, `oer_manifest.py` | Coverage | New/modified modules below 90% coverage | Add unit tests to reach ≥90% per policy | Repo policy requires ≥90% for new logic | Pytest coverage: 54%–72% for listed modules |
| Major | `scripts/dev_tools/atomic_executor/cli.py`, `scripts/dev_tools/fix_all.py`, `ck12_enrichment.py`, `ck12_catalog.py` | Line count | Files exceed 500-line cap | Split into smaller modules | Policy requires files ≤500 lines | `wc -l` shows 2389, 1132, 841, 521 lines |
| Minor | — | Baseline coverage | Baseline not documented | Capture baseline before further changes | Policy expects baseline for regression comparisons | Audit could not find baseline metric |
| Minor | Feature verification | Acceptance criteria require live HTTP checks and pipeline runs | Run end-to-end validation commands | Ensures manifest URLs and pipeline behavior are correct | Feature audit marks items UNVERIFIED/PARTIAL |

## Typed Python Audit

- **Pyright:** ✅ 0 errors (strict). No new `Any` usage detected in spot checks.
- **Coverage gap:** Significant under-coverage in pipeline scripts indicates untested branches and reduces confidence.
- **Docstrings/comments:** Not fully audited across the large diff; no obvious violations found in sampled modules.

## Test Quality Audit

- **Pytest:** ✅ 1362 tests passed in 7.12s with 87% coverage.
- **Pester:** ✅ 196 passed, 7 skipped; coverage 77.38% commands.
- **Bash:** ❌ 17 tests with 3 failures (`test_agent_mvp.bats`).

## Security/Correctness Checks

- No new secret material detected in the reviewed scope.
- Subprocess usage appears consistent with existing patterns, but not fully audited for all new scripts.

## Go/No-Go Recommendation

**No-Go (Needs Revision).** Merge readiness is blocked by shell test failures, coverage shortfalls for new modules, and file-size policy violations.

---

### Remediation Plan Prompt (atomic_planner)

Use this prompt verbatim with `atomic_planner`:

```
You are atomic_planner. Read the remediation inputs at:
/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/remediation-inputs.2026-01-29T05-45.md

Create and WRITE a remediation plan to:
/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/remediation-plan.2026-01-29T05-45.md

Requirements:
- Break work into atomic tasks with clear acceptance criteria.
- Include a final QA phase: format → lint → type-check → test (all applicable languages).
- Keep scope strictly to remediation inputs (no scope creep).
```

# Staged Code Review: Populate OpenStax/CK-12 Manifest

**Branch:** `feature/populate-open-stax-ck-12-manifest-#73`
**Date:** 2026-01-08

---

## 1. Executive Summary

This set of changes improves the robustness of the CK-12 catalog ingestion by switching from brittle HTML scraping to a discovered JSON API and adding browser-mimicking headers to avoid 403 Forbidden errors.

**Top 3 Risks:**
1.  **API Stability:** Reliance on an undocumented JSON endpoint (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) carries some risk if CK-12 changes their frontend architecture, though it is likely more stable than HTML scraping.
2.  **Type Safety (Python):** The JSON response is typed as `dict[str, Any]` and cast to `list[dict[str, object]]`. While necessary for external API usage, strict validation of the inner structure is minimal (checks for `books` list and `Content_URL`), relying on `get` defaults.
3.  **Script Usage:** The `download-ck12-bundle.ps1` script is a development helper that requires manual execution and network access.

**Recommendation:** **GO**
All components pass repo policy. Python changes are fully tested. PowerShell script is in correct location with unit tests.

---

## 2. Findings

| Severity | File | Location | Finding | Recommendation | Rationale | Evidence |
|----------|------|----------|---------|----------------|-----------|----------|
| **Info** | `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` | Line 106 | Loose typing on `title_raw` | Consider validating `title_raw` is a string explicitly before assignment, though the ternary handles it. | Purely stylistic; current code is safe but could be more explicit. | `title: str = title_raw if isinstance(title_raw, str) else slug` |

---

## 3. Typed Python Audit

- **No `Any` without justification:** The JSON return uses `dict[str, Any]` which is standard for `requests.json()`. A list cast uses `type: ignore[assignment]` which is acceptable here as the structure is externally defined.
- **Precise types:** Used `list[dict[str, object]]` instead of raw list. Good.
- **Error Types:** explicitly catches `requests.JSONDecodeError` and `requests.Timeout`. Excellent.
- **Exports:** Functions are module-level.

## 4. Test Quality Audit

- **Tests:** 7 tests cover the new JSON parsing logic.
- **Mocking:** `monkeypatch` used correctly for `requests.get`.
- **Coverage:** Happy path, missing title, empty list, and header regression test (Issue #73) included.

## 5. Security & Correctness

- **Subprocess:** No subprocess calls in Python code.
- **Secrets:** No secrets detected.
- **Inputs:** JSON is parsed safely; URL slugs are sanitized via existing helpers.

---

## 6. Remediation Plan

**Triggers:**
- `PolicyAudit.md` status is **PARTIAL**.
- `PSScriptAnalyzer` failed on `scripts/dev_tools/download-ck12-bundle.ps1`.

**Action:**
- Run `atomic_planner` to fix the PowerShell script compliance issues.

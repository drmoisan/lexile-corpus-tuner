# Policy Compliance Audit: sync-agents-from-instructions.ps1

**Audit Date:** 2025-12-11  
**Test File:** `tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1`  
**Code Under Test:** `scripts/dev-tools/sync-agents-from-instructions.ps1`  
**Total Tests:** 4  
**Test Result:** ❌ Failures in repo test suite unrelated to new tests (see Toolchain Results)

## Executive Summary
- New Pester tests cover YAML frontmatter stripping, missing-instruction errors, and AGENTS.md assembly.
- Formatting and PSScriptAnalyzer now pass with UTF-8 BOM applied to both script and tests.
- PoshQCTest failed because existing repository tests require `lpass` and Windows-style drives `C:/` and `D:/`; these dependencies are outside the new test scope.

## Toolchain Results
- **Formatting:** ✅ `Invoke-PoshQCFormat` (no findings)【0d30f3†L1-L1】
- **Analysis:** ✅ `Invoke-PoshQCAnalyze` (no findings)【0d30f3†L1-L1】
- **Testing:** ❌ `Invoke-PoshQCTest -Root .` failed: missing `lpass` command and drives `C:`/`D:` for existing tests (e.g., collect-commit-context, Convert-PoshQCCoverageToRelative).【3f8a09†L31-L41】【03019c†L1-L33】

## Policy Compliance

### General Code Change & PowerShell Code Change
- **PASS:** Code simplicity and separation of concerns maintained in both script and tests.
- **PASS:** Formatting and linting completed with no PSScriptAnalyzer findings.
- **FAIL:** Full toolchain test step due to external environment gaps (missing `lpass`, Windows drives).

### General Unit Test & PowerShell Unit Test
- **PASS:** Tests use Pester 5 with Describe/Context/It structure and isolated mocks for filesystem calls.
- **PASS:** Positive and negative scenarios covered for instruction parsing and AGENTS.md generation.
- **FAIL:** Repository-wide PoshQCTest failed because of unrelated legacy tests requiring unavailable dependencies; rerun when environment provides `lpass` and Windows drive mappings.

## Gaps and Follow-Ups
- Provision `lpass` CLI or skip legacy tests that depend on it when running PoshQCTest in this environment.
- Provide Windows-style drive mappings (`C:`, `D:`) or adjust legacy tests to tolerate Unix paths before rerunning the full suite.

## Recommendation
**Partially Compliant** – Merge after acknowledging external test failures. No policy blockers in the new script or tests; rerun full PoshQCTest once environment dependencies (`lpass`, Windows drive mapping) are available.

# PoshQc - Plan

- Issue: #21
- Owner: drmoisan
- Last Updated: 2025-12-09

## Required References (read, do not restate)

- Coding workflow and standards: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/powershell-code-change.instructions.md`
- Unit test policy: `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/powershell-unit-test.instructions.md`
- Tooling overview: `docs/developer-tooling.md`

**All work must comply with these policies; do not duplicate their content here.**

## Phases

### Phase 1: PoshQC module foundation [100%]
- [x] Scaffold PoshQC module (psd1/psm1) with formatter/analyzer/test entry points.
- [x] Vend settings (`pssa.settings.psd1`, `pester.runsettings.psd1`) and install helper.
- [x] Wire dev-tool wrappers to import module.

### Phase 2: Editor/tasks + docs [100%]
- [x] Update VS Code tasks to call PoshQC commands directly (format, analyze, autofix, test, install).
- [x] Update README and developer-tooling docs to point to PoshQC commands/settings.
- [x] Update PowerShell instructions to align with PoshQC.

### Phase 3: Validation and adoption [in progress]
- [ ] Run formatter/analyzer/tests on baseline scripts; resolve remaining PSSA findings (e.g., Write-Host, trailing whitespace, ShouldProcess).
- [ ] Validate on PowerShell 5.1 and 7.5+; document any compatibility gaps.
- [ ] Add Pester coverage for PoshQC functions (happy-path, error-path).
- [ ] Optional: document submodule/vendor guidance for reuse across repos.
- [ ] Add PoshQC format/analyze/test checks to CI so PowerShell changes are gated.

## Detailed Action Plan (tracked tasks)

### CI integration and gating
- [ ] Add PoshQC format/analyze/test jobs to `.github/workflows/ci.yml`, failing on any findings and emitting Pester JUnit under `artifacts/pester/pester-junit.xml`.
- [ ] Ensure CI installs tooling via `Install-PoshQCTools` on PS 7.5+; decide and document PS 5.1 matrix inclusion.

### PSSA hygiene and baseline readiness
- [ ] Run `Invoke-PoshQCAnalyze` repo-wide; fix all findings (e.g., Write-Host, ShouldProcess, validation attributes, trailing whitespace).
- [ ] Re-run formatter + analyzer to confirm clean baseline before adding/expanding tests.

### Compatibility validation
- [ ] Smoke PoshQC format/analyze/test on PS 5.1 and capture results/gaps.
- [ ] Smoke PoshQC format/analyze/test on PS 7.5+ and capture results/gaps.
- [ ] Add notes for any version-specific shims required.

### Pester coverage uplift (target ≥90% for PoshQC; raise repo ≥80%)
- [ ] Add Pester tests for PoshQC entry points:
	- [ ] `Invoke-PoshQCFormat` happy-path (formats sample file, no unintended edits).
	- [ ] `Invoke-PoshQCAnalyze` failure path (intentional violation triggers non-zero), success path (clean file passes).
	- [ ] `Invoke-PoshQCTest` runs configured Pester settings and writes JUnit to `artifacts/pester/pester-junit.xml`.
	- [ ] `Install-PoshQCTools` handles already-installed/no-op scenarios without errors.
- [ ] Add deterministic coverage for key dev-tool scripts where feasible (no temp files, no external calls), using mocks/fakes to stay isolated.
- [ ] Iterate until Pester coverage shows ≥90% for new/changed PowerShell modules and overall repo coverage trends toward ≥80%.

### Documentation and plan alignment
- [ ] Update this plan as tasks complete or scope changes; note CI matrix decisions and compatibility findings.
- [ ] Document any required shims or guidance discovered during compatibility and coverage work.

## Test Plan

- Unit: add Pester tests for PoshQC functions (formatter/analyzer/test invocations, settings resolution, per-file analyzer -Fix).
- Integration: run PoshQC tasks against repo scripts; confirm non-zero on findings and zero when clean.
- CLI/UX: verify tasks from VS Code plus direct CLI commands; ensure JUnit output in `artifacts/pester/pester-junit.xml`.
- Compatibility: smoke on 5.1 and 7.5+; ensure settings load and commands run.
- Edge: analyzer -Fix leaves remaining findings visible; formatter handles mixed line endings; excluded dirs respected.

## Open Questions / Notes

- Do we want to refactor legacy scripts to remove Write-Host and add ShouldProcess now or track separately?
- Should we publish PoshQC to PSGallery or keep vendored/submodule-only?
- Capture a minimal Pester suite to guard regressions in PoshQC entry points.


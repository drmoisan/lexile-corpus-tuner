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


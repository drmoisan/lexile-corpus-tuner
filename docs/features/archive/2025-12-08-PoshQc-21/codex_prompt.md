# Codex Execution Prompt — PoshQC + Dev-Tools Remediation

You are Codex running in a web execution environment. Fully execute the remediation plan in `docs/features/active/PoshQC/remediation-plan.md` for ALL PowerShell scripts/tests (PoshQC and non-PoshQC). Work autonomously, respecting repo policies and toolchain.

## Policies (must read/obey)
- `.github/instructions/general-code-change.instructions.md`
- `.github/instructions/general-unit-test.instructions.md`
- `.github/instructions/powershell-unit-test.instructions.md`
- Pester runsettings: `scripts/powershell/PoshQC/settings/pester.runsettings.psd1`
- PSScriptAnalyzer settings: `scripts/powershell/PoshQC/settings/pssa.settings.psd1`

## Scope
- PoshQC module: `scripts/powershell/PoshQC/PoshQC.psm1`, `convert-poshqc-coverage.ps1`, related tests under `tests/powershell/PoshQC*.Tests.ps1`.
- Non-PoshQC scripts under `scripts/dev-tools/*.ps1`, `scripts/powershell/**/*.ps1` (excluding PoshQC), with matching tests under `tests/powershell/**` and `tests/scripts/**`.
- Follow the master plan phases 0–10 in `docs/features/active/PoshQC/remediation-plan.md`.

## Execution Order (high level)
1) Phase 0: Read policies, scripts, tests; map dependencies. Do NOT modify code here.
2) Phase 1 & 6: Capture failing tests/coverage for both PoshQC and non-PoshQC (Invoke-PoshQCTest). Log failures.
3) Phase 2 & 7: Design injection/testability decisions; document deterministic ordering rules.
4) Phase 3: Implement PoshQC refactors per tasks P3-T1..P3-T35.
5) Phase 8: Implement non-PoshQC refactors per tasks P8-T1..P8-T15.
6) Phase 4: Add/adjust PoshQC tests per tasks P4-T1..P4-T26.
7) Phase 9: Add/adjust non-PoshQC tests per tasks P9-T1..P9-T36.
8) Phase 5 & 10: Toolchain: PoshQC formatter/analyzer, then Invoke-PoshQCTest with coverage (>90%); update docs.

## Required Behaviors
- Enforce dependency injection for external commands (git, actionlint, perl, lpass, PSScriptAnalyzer, Pester), filesystem I/O, env vars.
- Ensure deterministic ordering (sorting paths), early-return paths for empty inputs, explicit error paths for missing dependencies/files.
- Tests must be deterministic, isolated, no real external services, no temp files. Use mocks/stubs and injected delegates.
- Keep files cohesive and under 500 lines; minimal comments for non-obvious injections.

## Toolchain Commands (run until clean)
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`

## Deliverables
- Refactored scripts with injectable dependencies and deterministic behaviors.
- Expanded tests covering all scenarios listed in Phases 4 and 9; coverage >90% for touched modules.
- Updated documentation/plan notes reflecting completed tasks and any residual risks.

## Reporting
- Keep a log of failing tests before/after fixes.
- Summarize design decisions for injections and ordering rules.
- Confirm final toolchain pass success (format, analyze, test) with no failures.

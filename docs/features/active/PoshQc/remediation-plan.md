# PoshQC Remediation Plan

## Overview
Plan to fix failing Pester tests in `tests/powershell/PoshQC.Tests.ps1`, refactor `scripts/powershell/PoshQC/PoshQC.psm1` for dependency injection and isolation, eliminate policy violations, and raise coverage above 90% with scenario-driven tests.

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Read `.github/instructions/general-unit-test.instructions.md` and capture key constraints (independence, isolation, no temp files, determinism) in an internal note for this effort.
- [x] [P0-T2] Read `.github/instructions/powershell-unit-test.instructions.md` and capture key constraints (Pester v5, repo runsettings, organization, mocking guidance) in the same note.
- [x] [P0-T3] Review `scripts/powershell/PoshQC/PoshQC.psm1` and list each function, its purpose, and external dependencies (filesystem, PS modules, network, logging).
- [x] [P0-T4] Review `tests/powershell/PoshQC.Tests.ps1` and existing PoshQC feature docs (`plan.md`, `test-remediation-plan.md`) to understand current coverage and expectations.
- [x] [P0-T5] Read all files recursively under `scripts/` (dev-tools, powershell) and note purpose plus external dependencies for each script (git, actionlint, perl, lpass, filesystem, env vars).
- [x] [P0-T6] Read all tests under `tests/powershell` and `tests/scripts` and map each test file to the script it covers, noting any skip conditions or external dependency usage.

**Phase 1 — Current State & Failure Capture**
- [x] [P1-T1] Run `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."` and record failing tests with error messages and stack traces in a short log file.
- [x] [P1-T2] Generate current PoshQC code coverage (using existing Pester runsettings) and record per-function coverage percentages to pinpoint gaps relative to the 90% target.
- [x] [P1-T3] Document policy violations observed in current tests (e.g., missing edge scenarios, reliance on real filesystem/network, unclear assertions) in the work log.

**Phase 2 — Injection & Testability Design**
- [ ] [P2-T1] Draft a dependency map showing for each PoshQC function which external commands need wrappers/injection (Resolve-Path, Get-ChildItem, Set-Content, PSScriptAnalyzer, Pester, PSRepository/module install commands, logging).
- [ ] [P2-T2] Define a set of injectable delegates/parameters (with defaults) for filesystem access, module presence checks, module import, formatting/analyzer execution, and logging; record in a short design note to keep refactors consistent.
- [ ] [P2-T3] Decide on deterministic ordering/normalization rules for file lists and coverage paths to keep tests stable; note decisions in the design note.

**Phase 3 — Code Refactors & Behavioral Fixes**
- [x] [P3-T1] Add injectable path resolver to `Get-PoshQCFileList` with default bound to `Resolve-Path` and keep existing resolution semantics.
- [x] [P3-T2] Add injectable file enumerator to `Get-PoshQCFileList` with default bound to `Get-ChildItem` (recursive, PowerShell extensions) while preserving current include filters.
- [x] [P3-T3] Add injectable exclusion predicate to `Get-PoshQCFileList` so tests can simulate skip logic without touching the filesystem; keep default using `ExcludeDirs`.
- [x] [P3-T4] Enforce deterministic ordering in `Get-PoshQCFileList` output (e.g., sort by full path) after injection hooks are applied.
- [x] [P3-T5] Emit a controlled error from `Get-PoshQCFileList` when path resolution fails using the injected resolver (no silent fallbacks).
- [x] [P3-T6] Add injectable extension filter to `Get-PoshQCFileList` to drop non-PowerShell files while matching current default extensions.
- [x] [P3-T7] Add injectable TLS enforcement hook in `Install-PoshQCTool` so tests can assert TLS handling without changing system state.
- [x] [P3-T8] Add injectable PSRepository provider (get/register/set) in `Install-PoshQCTool` with defaults bound to current cmdlets.
- [x] [P3-T9] Add injectable module inventory checker in `Install-PoshQCTool` to detect already-installed modules and short-circuit installs.
- [x] [P3-T10] Add injectable module installer in `Install-PoshQCTool` that can be mocked to simulate success/failure paths deterministically.
- [x] [P3-T11] Add injectable info/warning logger in `Install-PoshQCTool` for PSGallery state changes and install outcomes.
- [x] [P3-T12] Ensure `Install-PoshQCTool` throws deterministic errors when the injected installer fails to add a required module.
- [x] [P3-T13] Add injectable module presence checker/importer to `Invoke-PoshQCFormat` to gate formatter execution.
- [x] [P3-T14] Add injectable settings-path existence check to `Invoke-PoshQCFormat` to raise the current error when missing.
- [x] [P3-T15] Add injectable formatter delegate to `Invoke-PoshQCFormat` to allow deterministic formatting outcomes in tests.
- [x] [P3-T16] Add injectable file reader/writer hooks to `Invoke-PoshQCFormat` preserving normalization logic.
- [x] [P3-T17] Add early return with injected logger in `Invoke-PoshQCFormat` when the injected file enumerator returns empty.
- [x] [P3-T18] Add injectable module presence checker/importer to `Invoke-PoshQCAnalyze` before analyzer invocation.
- [x] [P3-T19] Add injectable settings-path existence check to `Invoke-PoshQCAnalyze` to raise the current error when missing.
- [x] [P3-T20] Add injectable analyzer delegate to `Invoke-PoshQCAnalyze` to control findings and errors in tests.
- [x] [P3-T21] Add early return with injected logger in `Invoke-PoshQCAnalyze` when the injected file enumerator returns empty.
- [x] [P3-T22] Ensure `Invoke-PoshQCAnalyze` surfaces analyzer failures with file context using the injected delegate’s exceptions.
- [x] [P3-T23] Add injectable reader/writer hooks to `Convert-PoshQCCoverageToRelative` so tests can run entirely in-memory.
- [x] [P3-T24] Add injectable logger to `Convert-PoshQCCoverageToRelative` for skip/emit messages.
- [x] [P3-T25] Guard `Convert-PoshQCCoverageToRelative` to return early with info when both `InputPath` and `InputContent` are absent.
- [x] [P3-T26] Add injectable output-path resolver in `Convert-PoshQCCoverageToRelative` to derive default `.koverage.xml` targets deterministically.
- [x] [P3-T27] Ensure `Convert-PoshQCCoverageToRelative` honors `-PassThru` to skip writes and return converted content via injected hooks.
- [x] [P3-T28] Add injectable module checker/importer to `Invoke-PoshQCTest` to guard execution when Pester is missing.
- [x] [P3-T29] Add injectable settings loader in `Invoke-PoshQCTest` to import runsettings deterministically and fail fast when missing.
- [x] [P3-T30] Add injectable run-path expander and `ExcludePath` merger in `Invoke-PoshQCTest` to keep path handling deterministic.
- [x] [P3-T31] Add injectable test enumerator in `Invoke-PoshQCTest` to locate `*.Tests.ps1` files with ExcludeDirs honored.
- [x] [P3-T32] Add early return with injected logger in `Invoke-PoshQCTest` when no test files are found.
- [x] [P3-T33] Add injectable Pester invoker in `Invoke-PoshQCTest` so tests can simulate run success/failure without executing Pester.
- [x] [P3-T34] Add injectable coverage-copy hook in `Invoke-PoshQCTest` to validate Koverage export behavior with and without `-DisableKoverageCopy`.
- [x] [P3-T35] Wire new helper parameters into `Export-ModuleMember` and alias definitions to preserve the public surface with default injections.

**Phase 4 — Test Expansion (Scenario-Driven, >90% Coverage)**
- [ ] [P4-T1] Add Pester test for `Get-PoshQCFileList` returning an empty array when the injected file enumerator yields no PowerShell files in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T2] Add Pester test for `Get-PoshQCFileList` excluding files under directories listed in `ExcludeDirs` via injected exclusion predicate in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T3] Add Pester test for `Get-PoshQCFileList` throwing the expected error when root resolution fails in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T4] Add Pester test for `Get-PoshQCFileList` filtering out non-PowerShell extensions using an injected predicate in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T5] Add Pester test for `Get-PoshQCFileList` resolving relative roots to absolute paths using the injected resolver in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T6] Add Pester test for `Install-PoshQCTool` registering PSGallery when absent using injected repository provider in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T7] Add Pester test for `Install-PoshQCTool` setting repository to Trusted when untrusted and emitting warning on failure using injected hooks in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T8] Add Pester test for `Install-PoshQCTool` skipping installs when required modules are already present via injected module inventory in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T9] Add Pester test for `Install-PoshQCTool` throwing when an injected install operation fails in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T10] Add Pester test for `Invoke-PoshQCFormat` throwing when PSScriptAnalyzer is missing using injected module checker in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T11] Add Pester test for `Invoke-PoshQCFormat` throwing when the settings file is missing using injected file-exists hook in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T12] Add Pester test for `Invoke-PoshQCFormat` logging info and returning when file list is empty using injected logger in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T13] Add Pester test for `Invoke-PoshQCFormat` writing formatted content when the formatter changes text using injected file IO and formatter in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T14] Add Pester test for `Invoke-PoshQCFormat` performing no writes when content is unchanged using injected file IO and formatter in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T15] Add Pester test for `Invoke-PoshQCAnalyze` throwing when PSScriptAnalyzer is missing via injected module checker in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T16] Add Pester test for `Invoke-PoshQCAnalyze` throwing when settings file is missing via injected file-exists hook in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T17] Add Pester test for `Invoke-PoshQCAnalyze` logging info and returning when file list is empty via injected logger in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T18] Add Pester test for `Invoke-PoshQCAnalyze` throwing after analyzer returns findings via injected analyzer results in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T19] Add Pester test for `Convert-PoshQCCoverageToRelative` returning without writing when `InputPath` is missing via injected logger in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T20] Add Pester test for `Convert-PoshQCCoverageToRelative` deriving default output path when only `InputPath` is provided using injected reader/writer in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T21] Add Pester test for `Convert-PoshQCCoverageToRelative` converting paths correctly with trailing separator roots when `-PassThru` is used in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T22] Add Pester test for `Invoke-PoshQCTest` throwing when Pester is missing via injected module checker in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T23] Add Pester test for `Invoke-PoshQCTest` throwing when the settings file is missing via injected file-exists hook in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T24] Add Pester test for `Invoke-PoshQCTest` expanding `Run.Path` and merging `ExcludeDirs` into `ExcludePath` deterministically using injected test enumerator in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T25] Add Pester test for `Invoke-PoshQCTest` logging info and returning when no test files are found via injected logger in `tests/powershell/PoshQC.Tests.ps1`.
- [ ] [P4-T26] Add Pester test for `Invoke-PoshQCTest` invoking coverage copy when coverage is enabled and `-DisableKoverageCopy` is not set using injected coverage hook in `tests/powershell/PoshQC.Tests.ps1`.

**Phase 5 — Toolchain & Documentation**
- [ ] [P5-T1] Run PoshQC formatter and analyzer tasks (`Invoke-PoshQCFormat`, `Invoke-PoshQCAnalyze`) against the PowerShell files to ensure policy compliance after changes.
- [ ] [P5-T2] Run Pester via `Invoke-PoshQCTest -Root .` with coverage enabled, confirming >90% coverage for `PoshQC.psm1` and capturing the report artifact.
- [ ] [P5-T3] Update relevant feature docs (e.g., `plan.md`, `test-remediation-plan.md`) with the completed tasks and coverage results, noting any deviations from policies.

**Phase 6 — Non-PoshQC Inventory & Failures**
- [x] [P6-T1] Enumerate all non-PoshQC scripts under `scripts/dev-tools` and `scripts/powershell` with their external dependencies (git, actionlint, perl, lpass, filesystem paths, env vars).
- [x] [P6-T2] Map tests in `tests/powershell/**/*.Tests.ps1` and `tests/scripts/**/*.Tests.ps1` to their target scripts, noting skips and external dependency use.
- [x] [P6-T3] Run `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."` and log failing non-PoshQC tests with stack traces.
- [x] [P6-T4] Capture current coverage for `scripts/dev-tools/*.ps1` and `scripts/powershell/**/*.ps1` (excluding PoshQC) from the Pester report, listing functions below 90% coverage.

**Phase 7 — Non-PoshQC Testability & Injection Design**
- [ ] [P7-T1] Draft dependency maps per non-PoshQC script showing external commands, environment inputs, and filesystem touches that need injection or wrapping.
- [ ] [P7-T2] Define shared test helpers (mock runners, in-memory file maps, clock/random seeds) to replace external calls across dev-tools tests.
- [ ] [P7-T3] Specify deterministic ordering/normalization rules for file enumerations and output paths to stabilize assertions across scripts.

**Phase 8 — Non-PoshQC Script Refactors**
- [ ] [P8-T1] `collect-commit-context.ps1`: inject file writer and command runner; guard git failures with deterministic errors.
- [ ] [P8-T2] `collect-commit-context.ps1`: add allow-fail placeholder path using injected logger/writer without filesystem writes.
- [ ] [P8-T3] `collect-pull-request-context.ps1`: inject diff provider and numstat parser inputs; isolate brace-rename formatting logic.
- [ ] [P8-T4] `collect-pull-request-context.ps1`: add malformed/empty diff handling with deterministic errors.
- [ ] [P8-T5] `fix-all.ps1`: inject command runner and logger; ensure non-zero exits propagate deterministically.
- [ ] [P8-T6] `link-feature-docs.ps1`: inject file reader/writer; ensure append/replace behavior is deterministic when section missing.
- [ ] [P8-T7] `link-parent-child.ps1`: inject input provider and error logger; make validation paths testable without stdin.
- [ ] [P8-T8] `new-active-feature-folder.ps1`: inject filesystem creator and template loader; normalize checklist deterministically.
- [ ] [P8-T9] `new-potential-entry.ps1`: inject short-name pattern provider/validator; expose validation outcomes without filesystem.
- [ ] [P8-T10] `potential-to-issue.ps1`: inject content reader/writer and section extractor; guard missing-section errors deterministically.
- [ ] [P8-T11] `run-actionlint.ps1`: inject binary locator and command runner; add deterministic errors for missing binary and non-zero exit codes.
- [ ] [P8-T12] `run-cloc.ps1`: finalize injection for cloc exe/script selection and perl detection; ensure deterministic errors for missing binaries.
- [ ] [P8-T13] `sync-agents-from-instructions.ps1`: inject git status/diff provider and file writer; guard missing instruction files with deterministic errors.
- [ ] [P8-T14] `tree.ps1`: inject file enumerator/attribute provider; make hidden/exclude handling deterministic without real filesystem.
- [ ] [P8-T15] `scripts/powershell/convert-poshqc-coverage.ps1` (if present): inject reader/writer and repo-root resolver; honor pass-thru/skip paths without disk access.

**Phase 9 — Non-PoshQC Test Remediation & Coverage**
- [ ] [P9-T1] Add tests for `collect-commit-context.ps1` covering success capture via injected writer/runner.
- [ ] [P9-T2] Add tests for `collect-commit-context.ps1` covering allow-fail placeholder emission.
- [ ] [P9-T3] Add tests for `collect-commit-context.ps1` covering git failure path with deterministic error.
- [ ] [P9-T4] Add tests for `collect-pull-request-context.ps1` covering brace-rename formatting.
- [ ] [P9-T5] Add tests for `collect-pull-request-context.ps1` covering numstat parsing totals/file list.
- [ ] [P9-T6] Add tests for `collect-pull-request-context.ps1` covering extension summary counts.
- [ ] [P9-T7] Add tests for `collect-pull-request-context.ps1` covering issue reference extraction.
- [ ] [P9-T8] Add tests for `collect-pull-request-context.ps1` covering malformed/empty diff handling.
- [ ] [P9-T9] Add tests for `fix-all.ps1` covering step/success/failure logging with injected runner.
- [ ] [P9-T10] Add tests for `fix-all.ps1` covering non-zero exit propagation.
- [ ] [P9-T11] Add tests for `link-feature-docs.ps1` covering section replace behavior.
- [ ] [P9-T12] Add tests for `link-feature-docs.ps1` covering append-when-missing behavior.
- [ ] [P9-T13] Add tests for `link-parent-child.ps1` covering trimmed inputs with injected input provider.
- [ ] [P9-T14] Add tests for `link-parent-child.ps1` covering missing-input validation path.
- [ ] [P9-T15] Add tests for `new-active-feature-folder.ps1` covering checklist normalization.
- [ ] [P9-T16] Add tests for `new-active-feature-folder.ps1` covering section extraction/replacement.
- [ ] [P9-T17] Add tests for `new-active-feature-folder.ps1` covering placeholder substitution.
- [ ] [P9-T18] Add tests for `new-potential-entry.ps1` covering valid short-name acceptance via injected validator.
- [ ] [P9-T19] Add tests for `new-potential-entry.ps1` covering invalid short-name rejection.
- [ ] [P9-T20] Add tests for `potential-to-issue.ps1` covering section extraction.
- [ ] [P9-T21] Add tests for `potential-to-issue.ps1` covering metadata line insertion.
- [ ] [P9-T22] Add tests for `potential-to-issue.ps1` covering missing-section error path.
- [ ] [P9-T23] Add tests for `run-actionlint.ps1` covering missing binary error via injected locator.
- [ ] [P9-T24] Add tests for `run-actionlint.ps1` covering non-zero exit handling via injected runner.
- [ ] [P9-T25] Add tests for `run-cloc.ps1` covering cloc.exe preference on Windows via injected binaries.
- [ ] [P9-T26] Add tests for `run-cloc.ps1` covering perl fallback when cloc.exe absent.
- [ ] [P9-T27] Add tests for `run-cloc.ps1` covering missing binary error path.
- [ ] [P9-T28] Add tests for `run-cloc.ps1` covering custom path parameter handling.
- [ ] [P9-T29] Add tests for `sync-agents-from-instructions.ps1` covering missing instruction file error.
- [ ] [P9-T30] Add tests for `sync-agents-from-instructions.ps1` covering successful sync with injected content/writer.
- [ ] [P9-T31] Add tests for `sync-agents-from-instructions.ps1` covering git-dirty detection via injected provider.
- [ ] [P9-T32] Add tests for `tree.ps1` covering exclusions and hidden-entry toggle with injected enumerator.
- [ ] [P9-T33] Add tests for `tree.ps1` covering directories-only toggle.
- [ ] [P9-T34] Add tests for `convert-poshqc-coverage.ps1` covering missing input skip with injected logger.
- [ ] [P9-T35] Add tests for `convert-poshqc-coverage.ps1` covering pass-thru returning converted content without writes.
- [ ] [P9-T36] Add tests for `convert-poshqc-coverage.ps1` covering default output path derivation for `.koverage.xml`.

**Phase 10 — Non-PoshQC Policy Compliance**
- [ ] [P10-T1] Verify refactored non-PoshQC scripts stay under 500 lines and remain cohesive; add minimal comments for non-obvious injections.
- [ ] [P10-T2] Run PoshQC formatter/analyzer on modified non-PoshQC scripts/tests and fix all findings.
- [ ] [P10-T3] Run `Invoke-PoshQCTest -Root .` with coverage enabled to confirm >90% coverage for modified non-PoshQC scripts and absence of external dependency calls during tests.
- [ ] [P10-T4] Update docs (existing plan/test-remediation notes) with completed non-PoshQC tasks, coverage results, and any residual risks.

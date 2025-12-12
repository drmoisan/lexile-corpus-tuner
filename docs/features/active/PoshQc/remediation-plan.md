# PoshQC Remediation Plan

## Overview
Plan to fix failing Pester tests in `tests/powershell/PoshQC.Tests.ps1`, refactor `scripts/powershell/PoshQC/PoshQC.psm1` for dependency injection and isolation, eliminate policy violations, and raise coverage above 90% with scenario-driven tests.

**Phase 0 — Context & Inputs**
- [ ] [P0-T1] Read `.github/instructions/general-unit-test.instructions.md` and capture key constraints (independence, isolation, no temp files, determinism) in an internal note for this effort.
- [ ] [P0-T2] Read `.github/instructions/powershell-unit-test.instructions.md` and capture key constraints (Pester v5, repo runsettings, organization, mocking guidance) in the same note.
- [ ] [P0-T3] Review `scripts/powershell/PoshQC/PoshQC.psm1` and list each function, its purpose, and external dependencies (filesystem, PS modules, network, logging).
- [ ] [P0-T4] Review `tests/powershell/PoshQC.Tests.ps1` and existing PoshQC feature docs (`plan.md`, `test-remediation-plan.md`) to understand current coverage and expectations.

**Phase 1 — Current State & Failure Capture**
- [ ] [P1-T1] Run `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."` and record failing tests with error messages and stack traces in a short log file.
- [ ] [P1-T2] Generate current PoshQC code coverage (using existing Pester runsettings) and record per-function coverage percentages to pinpoint gaps relative to the 90% target.
- [ ] [P1-T3] Document policy violations observed in current tests (e.g., missing edge scenarios, reliance on real filesystem/network, unclear assertions) in the work log.

**Phase 2 — Injection & Testability Design**
- [ ] [P2-T1] Draft a dependency map showing for each PoshQC function which external commands need wrappers/injection (Resolve-Path, Get-ChildItem, Set-Content, PSScriptAnalyzer, Pester, PSRepository/module install commands, logging).
- [ ] [P2-T2] Define a set of injectable delegates/parameters (with defaults) for filesystem access, module presence checks, module import, formatting/analyzer execution, and logging; record in a short design note to keep refactors consistent.
- [ ] [P2-T3] Decide on deterministic ordering/normalization rules for file lists and coverage paths to keep tests stable; note decisions in the design note.

**Phase 3 — Code Refactors & Behavioral Fixes**
- [ ] [P3-T1] Add injectable path resolver to `Get-PoshQCFileList` with default bound to `Resolve-Path` and keep existing resolution semantics.
- [ ] [P3-T2] Add injectable file enumerator to `Get-PoshQCFileList` with default bound to `Get-ChildItem` (recursive, PowerShell extensions) while preserving current include filters.
- [ ] [P3-T3] Add injectable exclusion predicate to `Get-PoshQCFileList` so tests can simulate skip logic without touching the filesystem; keep default using `ExcludeDirs`.
- [ ] [P3-T4] Enforce deterministic ordering in `Get-PoshQCFileList` output (e.g., sort by full path) after injection hooks are applied.
- [ ] [P3-T5] Emit a controlled error from `Get-PoshQCFileList` when path resolution fails using the injected resolver (no silent fallbacks).
- [ ] [P3-T6] Add injectable extension filter to `Get-PoshQCFileList` to drop non-PowerShell files while matching current default extensions.
- [ ] [P3-T7] Add injectable TLS enforcement hook in `Install-PoshQCTool` so tests can assert TLS handling without changing system state.
- [ ] [P3-T8] Add injectable PSRepository provider (get/register/set) in `Install-PoshQCTool` with defaults bound to current cmdlets.
- [ ] [P3-T9] Add injectable module inventory checker in `Install-PoshQCTool` to detect already-installed modules and short-circuit installs.
- [ ] [P3-T10] Add injectable module installer in `Install-PoshQCTool` that can be mocked to simulate success/failure paths deterministically.
- [ ] [P3-T11] Add injectable info/warning logger in `Install-PoshQCTool` for PSGallery state changes and install outcomes.
- [ ] [P3-T12] Ensure `Install-PoshQCTool` throws deterministic errors when the injected installer fails to add a required module.
- [ ] [P3-T13] Add injectable module presence checker/importer to `Invoke-PoshQCFormat` to gate formatter execution.
- [ ] [P3-T14] Add injectable settings-path existence check to `Invoke-PoshQCFormat` to raise the current error when missing.
- [ ] [P3-T15] Add injectable formatter delegate to `Invoke-PoshQCFormat` to allow deterministic formatting outcomes in tests.
- [ ] [P3-T16] Add injectable file reader/writer hooks to `Invoke-PoshQCFormat` preserving normalization logic.
- [ ] [P3-T17] Add early return with injected logger in `Invoke-PoshQCFormat` when the injected file enumerator returns empty.
- [ ] [P3-T18] Add injectable module presence checker/importer to `Invoke-PoshQCAnalyze` before analyzer invocation.
- [ ] [P3-T19] Add injectable settings-path existence check to `Invoke-PoshQCAnalyze` to raise the current error when missing.
- [ ] [P3-T20] Add injectable analyzer delegate to `Invoke-PoshQCAnalyze` to control findings and errors in tests.
- [ ] [P3-T21] Add early return with injected logger in `Invoke-PoshQCAnalyze` when the injected file enumerator returns empty.
- [ ] [P3-T22] Ensure `Invoke-PoshQCAnalyze` surfaces analyzer failures with file context using the injected delegate’s exceptions.
- [ ] [P3-T23] Add injectable reader/writer hooks to `Convert-PoshQCCoverageToRelative` so tests can run entirely in-memory.
- [ ] [P3-T24] Add injectable logger to `Convert-PoshQCCoverageToRelative` for skip/emit messages.
- [ ] [P3-T25] Guard `Convert-PoshQCCoverageToRelative` to return early with info when both `InputPath` and `InputContent` are absent.
- [ ] [P3-T26] Add injectable output-path resolver in `Convert-PoshQCCoverageToRelative` to derive default `.koverage.xml` targets deterministically.
- [ ] [P3-T27] Ensure `Convert-PoshQCCoverageToRelative` honors `-PassThru` to skip writes and return converted content via injected hooks.
- [ ] [P3-T28] Add injectable module checker/importer to `Invoke-PoshQCTest` to guard execution when Pester is missing.
- [ ] [P3-T29] Add injectable settings loader in `Invoke-PoshQCTest` to import runsettings deterministically and fail fast when missing.
- [ ] [P3-T30] Add injectable run-path expander and `ExcludePath` merger in `Invoke-PoshQCTest` to keep path handling deterministic.
- [ ] [P3-T31] Add injectable test enumerator in `Invoke-PoshQCTest` to locate `*.Tests.ps1` files with ExcludeDirs honored.
- [ ] [P3-T32] Add early return with injected logger in `Invoke-PoshQCTest` when no test files are found.
- [ ] [P3-T33] Add injectable Pester invoker in `Invoke-PoshQCTest` so tests can simulate run success/failure without executing Pester.
- [ ] [P3-T34] Add injectable coverage-copy hook in `Invoke-PoshQCTest` to validate Koverage export behavior with and without `-DisableKoverageCopy`.
- [ ] [P3-T35] Wire new helper parameters into `Export-ModuleMember` and alias definitions to preserve the public surface with default injections.

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

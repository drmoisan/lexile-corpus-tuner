# PoshQC Remediation Notes (Phases 0, 1, 6)

## Phase 0 – Context and Inventory
- **Policy constraints:**
  - General unit tests must be independent, deterministic, avoid external dependencies, and forbid temporary files; new modules target ≥90% coverage.【F:.github/instructions/general-unit-test.instructions.md†L12-L58】
  - PowerShell tests must use Pester v5 with the repo runsettings, mirror code layout, and stay compatible with PowerShell 5.1 and 7.5+.【F:.github/instructions/powershell-unit-test.instructions.md†L11-L38】
- **PoshQC module surface and dependencies:**
  - `Get-PoshQCFileList` resolves roots, enumerates PowerShell files, and skips default excluded directories via `Get-ChildItem` and `Resolve-Path`.【F:scripts/powershell/PoshQC/PoshQC.psm1†L10-L30】
  - `Install-PoshQCTool` toggles TLS, manages PSGallery trust/registration, and installs specific PSScriptAnalyzer/Pester versions, relying on PowerShellGet and network access.【F:scripts/powershell/PoshQC/PoshQC.psm1†L39-L83】
  - `Invoke-PoshQCFormat`/`Invoke-PoshQCAnalyze` import PSScriptAnalyzer, load settings, enumerate files, and perform formatting/analyzer passes with filesystem reads/writes.【F:scripts/powershell/PoshQC/PoshQC.psm1†L91-L152】
  - `Convert-PoshQCCoverageToRelative` resolves repo roots and coverage paths, normalizes separators, and writes a `.koverage.xml` copy when `-PassThru` is not used.【F:scripts/powershell/PoshQC/PoshQC.psm1†L187-L218】【F:scripts/powershell/PoshQC/PoshQC.psm1†L220-L243】
  - `Invoke-PoshQCTest` loads Pester/runsettings, expands paths, enumerates `*.Tests.ps1`, invokes Pester, and optionally copies coverage output via `Convert-PoshQCCoverageToRelative`.【F:scripts/powershell/PoshQC/PoshQC.psm1†L254-L332】【F:scripts/powershell/PoshQC/PoshQC.psm1†L334-L372】
- **Existing PoshQC tests/docs reviewed:** current tests only cover `Convert-PoshQCCoverageToRelative`; broader coverage lives in `PoshQC.Comprehensive.Tests.ps1` and entry-point tests referenced by the remediation plan.【F:tests/powershell/PoshQC.Tests.ps1†L1-L79】
- **Scripts inventory (non-PoshQC):**
  - Dev-tools scripts include Git/reporting helpers (`collect-commit-context.ps1`, `collect-pull-request-context.ps1`), linters (`run-actionlint.ps1`, `run-cloc.ps1`), automation (`fix-all.ps1`), doc linkers, and feature scaffolding scripts; they depend on git, external CLIs (actionlint, perl, poetry, gh), filesystem I/O, and environment variables.【F:scripts/dev-tools/collect-commit-context.ps1†L1-L60】【F:scripts/dev-tools/collect-pull-request-context.ps1†L1-L122】【F:scripts/dev-tools/run-actionlint.ps1†L14-L80】【F:scripts/dev-tools/run-cloc.ps1†L1-L84】【F:scripts/dev-tools/fix-all.ps1†L1-L96】
  - Additional PoshQC-adjacent helper script `convert-poshqc-coverage.ps1` shells into `Convert-PoshQCCoverageToRelative` after resolving repo paths.【F:scripts/powershell/PoshQC/convert-poshqc-coverage.ps1†L1-L32】
- **Test inventory mapping:**
  - Pester suites exist for each dev-tool script under `tests/scripts/dev-tools` and `tests/powershell/dev-tools.*` plus PoshQC entry-point suites under `tests/powershell/PoshQC.*`. Non-PoshQC scaffolding tests live in `tests/powershell/new-potential-entry.Tests.ps1` and `tests/powershell/dev-tools.Tests.ps1` for shared helpers.【F:tests/powershell/dev-tools.Tests.ps1†L1-L12】【F:tests/scripts/dev-tools/run-actionlint.Tests.ps1†L1-L6】

## Phase 1 – Current Failures and Coverage
- Ran `Invoke-PoshQCTest -Root .`; tests discovered 435 cases with 100 failures and 9 skips.【80c3c1†L1-L4】【54930d†L69-L72】
- Notable failing areas:
  - `run-cloc.ps1` path resolution and invocation mocks failing due to missing test fixtures and parameter binding issues.【72354d†L28-L60】【c1ebee†L25-L65】
  - `Convert-PoshQCCoverageToRelative` failures across multiple suites when repo roots or drives are absent.【72354d†L63-L71】【da5d66†L15-L28】【c1ebee†L11-L23】
  - `new-potential-entry.ps1` tests fail during `BeforeAll` because `Import-ScriptFunction` helper is not available.【6351f3†L6-L19】
  - `collect-commit-context.ps1` integration tests fail on drive resolution when invoking git-backed sections.【da5d66†L33-L47】
  - `run-actionlint.ps1` suite fails immediately because `Import-ScriptFunction` is missing.【c1ebee†L24-L32】
  - `Invoke-PoshQCTest` coverage-copy expectations unmet (mock not invoked).【af80d5†L17-L28】
  - `load-openai-key.ps1` fails when `lpass` command is unavailable.【72354d†L61-L63】【c1ebee†L1-L5】
- Coverage artifact (`artifacts/pester/powershell-coverage.xml`) was not created during the failing run; per-function percentages were unavailable for recording.

## Phase 6 – Non-PoshQC Inventory and Failures
- **Scripts and external dependencies:**
  - `collect-commit-context.ps1` relies heavily on git commands for repo state and writes to `artifacts/commit_context.txt`.【F:scripts/dev-tools/collect-commit-context.ps1†L31-L60】
  - `collect-pull-request-context.ps1` wraps git operations, parses diffs/numstat, formats summaries, and writes PR context output. It depends on git, filesystem paths, and optional user-supplied refs.【F:scripts/dev-tools/collect-pull-request-context.ps1†L1-L122】【F:scripts/dev-tools/collect-pull-request-context.ps1†L123-L214】
  - `run-actionlint.ps1` locates actionlint on PATH or downloads it from GitHub, uses `Invoke-WebRequest`, `Expand-Archive`, and adjusts `$env:PATH`.【F:scripts/dev-tools/run-actionlint.ps1†L14-L80】
  - `run-cloc.ps1` chooses between bundled `cloc.exe` or perl-backed script and requires git for repo-aware counts.【F:scripts/dev-tools/run-cloc.ps1†L28-L77】
  - `fix-all.ps1` orchestrates `poetry` commands for Black, Ruff, Pyright, and Pytest with retry logic; depends on Python tooling availability.【F:scripts/dev-tools/fix-all.ps1†L1-L83】
  - Other dev-tool scripts (linkers, tree, scaffolding, sync) manipulate markdown files, feature folders, or AGENTS.md using filesystem and git/gh interactions (see tests for behaviors).【F:tests/scripts/dev-tools/link-feature-docs.Tests.ps1†L1-L20】【F:tests/scripts/dev-tools/link-parent-child.Tests.ps1†L1-L23】
- **Test mapping:**
  - Each dev-tool script has paired suites under `tests/scripts/dev-tools/*.Tests.ps1`; additional behavioral suites under `tests/powershell/dev-tools.*` exercise helpers and shared scenarios.【F:tests/scripts/dev-tools/collect-pull-request-context.Tests.ps1†L1-L16】【F:tests/powershell/dev-tools.Tests.ps1†L1-L12】
- **Current failing non-PoshQC tests (from Phase 1 run):**
  - `run-cloc.ps1` multiple contexts failing due to path resolution and mock setup gaps.【72354d†L28-L60】【c1ebee†L25-L65】
  - `collect-commit-context.ps1` integration cases failing because git path resolution uses non-existent drives in test environment.【da5d66†L33-L47】
  - `run-actionlint.ps1` suite failing because shared `Import-ScriptFunction` helper is not located.【c1ebee†L24-L32】
  - `load-openai-key.ps1` failing without `lpass` executable.【72354d†L61-L63】【c1ebee†L1-L5】
- **Non-PoshQC coverage:** no coverage report emitted in the failing run; function-level coverage for dev-tools scripts remains unknown.

## Immediate Fix Plan (Phase 3 follow-up)
- Stabilize `Convert-PoshQCCoverageToRelative` by allowing string-based repo roots when the path cannot be resolved so XML-only tests no longer throw drive errors.
- Repair dev-tools test harnesses by fixing helper imports (`Import-ScriptFunction`) and correcting script paths for nested test folders so the helper can locate source scripts.
- Add deterministic, injectable path resolution and command execution to `run-cloc.ps1` to avoid hitting missing Windows drives, cloc/perl binaries, or the call operator during tests; update the paired tests to use the new injection points instead of real process launches.
- Record missing external tools from the latest test run (`lpass`, `cloc`/`perl`) for dependency reporting.

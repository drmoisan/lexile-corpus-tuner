# PoshQc — Spec

- Issue: #21
- Owner: drmoisan
- Last Updated: 2025-12-09

## Overview

Need a reusable, repo-agnostic PowerShell quality-control layer (formatter, lint, tests) so any custom PowerShell scripts across repos follow the same strict standards as Python (Black/Ruff/Pyright) without relying on Poetry/venv. Centralize settings, tasks, and install steps to reduce drift and friction.


## Behavior

- Ship a PoshQC module (vendored folder) with formatter (Invoke-Formatter), analyzer (PSScriptAnalyzer), Pester config/runner, and installer helper.
- Provide VS Code tasks that import PoshQC directly (format, analyze, autofix, test) and fail the toolchain on any findings.
- Enforce compatibility with PowerShell 5.1 and 7.5+, using strict analyzer settings.
- Keep all settings (PSSA, Pester) inside the module for reuse across repos; instructions reference PoshQC commands/tasks.


## Inputs / Outputs

- Inputs: PowerShell files under repo root; PoshQC settings at `scripts/powershell/PoshQC/settings/*.psd1`.
- CLI surface via tasks/commands:
  - `Install-PoshQCTool` (alias `Install-PoshQCTools`) imports PSGallery modules PSScriptAnalyzer 1.22.0 and Pester 5.6.1 in CurrentUser scope.
  - `Invoke-PoshQCFormat` (Invoke-Formatter with settings).
  - `Invoke-PoshQCAnalyze` (PSScriptAnalyzer, strict, non-zero on findings).
  - `Invoke-PoshQCTest` (Pester with repo config).
- Outputs:
  - Formatted PowerShell files (in-place).
  - Analyzer findings (console, non-zero exit).
  - Pester JUnit: `artifacts/pester/pester-junit.xml`.
  - VS Code tasks: `PoshQC: 1 format`, `PoshQC: 2 analyze`, `PoshQC: 2b autofix (PSSA -Fix)`, `PoshQC: 4 test (Pester)`, `Dev: Install PowerShell Tooling`.

## API / CLI Surface

Examples:

- Install tools:  
  `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Install-PoshQCTool"`
- Format:  
  `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."`
- Analyze:  
  `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."`
- Autofix (PSSA -Fix):  
  `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; $files = Get-ChildItem -Recurse -Include *.ps1, *.psm1 -File; foreach ($f in $files) { Invoke-ScriptAnalyzer -Path $f.FullName -Settings ./scripts/powershell/PoshQC/settings/pssa.settings.psd1 -Severity Error,Warning,Information -Fix } }"`
- Test (Pester):  
  `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`

## Data & State

- In-place edits to PowerShell files by formatter/autofix.
- Tool installs occur in CurrentUser scope (PSScriptAnalyzer, Pester).
- Test results written to `artifacts/pester/pester-junit.xml`.
- Settings are vendored inside `scripts/powershell/PoshQC/settings`.

## Constraints & Risks

- PSScriptAnalyzer `-Fix` only covers a subset of rules; manual fixes still required.
- Compatibility matrix limited to 5.1/7.5+; older hosts not supported.
- Write-Host and ShouldProcess enforcement may require refactoring legacy scripts.
- Module is vendored (not published to PSGallery); updates need coordinated bumps across repos.


## Definition of Done

- [x] Behavior matches acceptance criteria (except pending 5.1 validation)
- [ ] Tests updated/added (Pester suite for PoshQC helpers)
- [x] Docs updated (README, instructions, tasks)
- [ ] Telemetry/logging (if applicable)
- [ ] CI runs PoshQC format/analyze/test checks and gates PowerShell changes

## Seeded Test Conditions (from potential)
- [ ] PoshQC format: no errors; rewrites expected files; leaves others untouched.
- [ ] PoshQC analyze: fails with findings in a sample file, passes when clean.
- [ ] PoshQC autofix: applies available fixes and leaves remaining findings visible.
- [ ] PoshQC test: runs Pester config, emits JUnit to artifacts, honors repo root paths.
- [ ] Cross-version smoke: run format/analyze/test on PS 5.1 and 7.5+.
- [ ] Instructions/tasks point to correct paths and commands.

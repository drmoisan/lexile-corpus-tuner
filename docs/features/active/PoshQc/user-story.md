# `PoshQc` - User Story

- Issue: #21
- Owner: drmoisan
- Status: In Progress
- Last Updated: 2025-12-09

## Story Statement

- As a repo maintainer, I want a reusable PowerShell QC toolkit I can drop into any repo, so that PowerShell scripts meet the same strict standards as our Python toolchain.
- As a contributor, I want one-click tasks to format/lint/test PowerShell, so that I can ship changes without memorizing commands or fighting tool installs.

## Problem / Why

Need a reusable, repo-agnostic PowerShell quality-control layer (formatter, lint, tests) so any custom PowerShell scripts across repos follow the same strict standards as Python (Black/Ruff/Pyright) without relying on Poetry/venv. Centralize settings, tasks, and install steps to reduce drift and friction.


## Personas & Scenarios

- Persona: Repo Maintainer (PowerShell + Python repos)
  - Cares about consistent QC across languages and repos; wants low-friction onboarding.
  - Constraints: multi-repo, mixed environments (WinPS 5.1, PS 7.5+), minimal manual setup.
  - Goals: one module to import, common tasks, strict rules, zero drift.
- Scenario: Contributor adds a new PowerShell script
  - Trigger: new automation script added.
  - Steps: run `PoshQC: 1 format`, `PoshQC: 2 analyze`, optional `PoshQC: 2b autofix`, then `PoshQC: 4 test (Pester)`.
  - Obstacles: missing tooling (fixed by `Install-PoshQCTools`), analyzer findings (must be resolved).
  - Outcome: script passes formatter/analyzer/tests with enforced compatibility for 5.1/7.5+.


## Acceptance Criteria

- [x] PoshQC module provides `Install-PoshQCTools`, `Invoke-PoshQCFormat`, `Invoke-PoshQCAnalyze`, `Invoke-PoshQCTest` with strict settings.
- [x] VS Code tasks call PoshQC commands directly (no shims) for format, analyze, autofix, and Pester test.
- [x] Analyzer runs non-interactively and exits non-zero on any findings; autofix task applies available fixes.
- [x] Pester config uses repo-relative paths and writes JUnit output under artifacts.
- [x] PowerShell instructions updated to reference PoshQC commands, settings paths, and tasks.
- [ ] Compatibility validated on PowerShell 5.1 and 7.5+.
- [ ] PoshQC format/analyze/test checks are wired into CI so PowerShell changes are gated.


## Non-Goals

- Publish PoshQC to PSGallery (vendored only).
- Add PowerShell type-checking beyond PSScriptAnalyzer.
- Manage Python/Poetry tooling (remains separate). 


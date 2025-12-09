---
applyTo: "**/*.ps1"
name: powershell-unit-test-policy
description: "PowerShell-specific unit test rules, layered on top of the general unit test policy"
---

# PowerShell Unit Test Policy

This policy **extends** `general-unit-test.instructions.md` and applies to all PowerShell tests in this repo.

You must follow **both**:

- The general unit test policy, and
- The PowerShell-specific rules below.

If there is any conflict between these documents, halt and notify the user.

---

## 1. Framework and Scope

- **Testing framework:** All PowerShell tests must use **Pester** (v5.x).
- Use the repo config at `scripts/powershell/PoshQC/settings/pester.runsettings.psd1`. Run via PoshQC:
  - `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`
- VS Code task: `PoshQC: 4 test (Pester)`
- Keep tests compatible with Windows PowerShell 5.1 and PowerShell 7.5+.

---

## 2. Test Style and Structure (PowerShell)

- Name test files `*.Tests.ps1`.
- Organize tests with `Describe`/`Context`/`It`. One behavior per `It`.
- Prefer explicit Arrange/Act/Assert blocks; keep assertions specific and actionable.
- Avoid relying on global state; set up and tear down inside the test scope.
- Mock external calls only as needed to satisfy isolation and determinism; prefer real code paths for pure helpers.

---

## 3. Running the Toolchain (PowerShell Tests)

- When running the "After Making Changes" toolchain, the **testing step** for PowerShell must use:
  - `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`
- Do **not** substitute other test runners for PowerShell work without explicit approval.

This file defines **how** PowerShell tests are written and executed; the general code change policy defines **when** to run the toolchain and how strictly to enforce it.

---
applyTo: "**/*.ps1,**/*.psm1,**/*.psd1,**/*.ps1xml"   
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
- Keep tests compatible with PowerShell 7+.

---

## 2. Test Style and Structure (PowerShell)

- **Focused unit tests**

  - Write focused tests that exercise a single function, method, or behavior.
  - Prefer testing behavior directly over testing implementation details.

- **Mocking**

  - Use mocking sparingly. Prefer real code paths and pure functions where possible.
  - Only introduce mocks/stubs when needed to satisfy isolation, determinism and “avoid external dependencies” requirements (e.g., external services, heavy resources).

- **Organization**

  - Organize tests into folders in a way that mirrors the code under test (e.g., `tests/scripts/dev-tools/ScriptName.Tests.ps1` for `scripts/dev-tools/ScriptName.ps1`).

---

## 3. Naming and Readability (Python)

- **Naming conventions**

  - Name test files `*.Tests.ps1`.
  - Organize tests with `Describe`/`Context`/`It`. One behavior per `It`.
  - Group related tests logically within the same file or test class.

- **Docstrings and comments**

  - Where the intent is not obvious from the `Describe`/`Context`/`It` alone, include a short docstring or comment summarizing:
    - The scenario being tested.
    - The expected outcome.

---

## 4. Running the Toolchain (PowerShell Tests)

- When running the "After Making Changes" toolchain, the **testing step** for PowerShell must use:
  - `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`
- Do **not** substitute other test runners for PowerShell work without explicit approval.

This file defines **how** PowerShell tests are written and executed; the general code change policy defines **when** to run the toolchain and how strictly to enforce it.

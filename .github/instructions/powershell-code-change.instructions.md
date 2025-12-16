---
applyTo: "**/*.ps1,**/*.psm1,**/*.psd1,**/*.ps1xml"
name: powershell-code-change-policy
description: "PowerShell-specific code change rules layered on top of the general code change policy"
---
# PowerShell Code Change Policy

This policy **extends** `general-code-change.instructions.md` and applies to all PowerShell source, scripts, and modules (`*.ps1`, `*.psm1`) in this repo.

You must:

- Apply **all** rules in the general code change policy.
- Apply **all** PowerShell-specific rules in this file.
- Apply the unit test policies (`general-unit-test.instructions.md` and `powershell-unit-test.instructions.md`) for any PowerShell tests.

If you encounter any conflicting instructions, **halt and notify the user.**

---

## 1. Tooling & Baseline for PowerShell

1) **Formatting - Invoke-Formatter**

- Format all PowerShell files using the PoshQC formatter (Invoke-Formatter). Example:
  - `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."`
- VS Code task: `PoshQC: 1 format`
- Do not hand-format; re-run the formatter whenever PSScriptAnalyzer would change whitespace/indentation.

2) **Linting - PSScriptAnalyzer**

- Run the PoshQC analyzer (PSScriptAnalyzer) with repo settings. Example:
  - `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."`
- VS Code task: `PoshQC: 2 analyze`
- An optional autofix task is available: `PoshQC: 2b autofix (PSSA -Fix)`; review diffs after running.
- Fix **all** findings (Error/Warning/Information). No rule suppressions unless strictly necessary and localized with a comment.

3) **Compatibility**

- Keep scripts compatible with **PowerShell 7+** (enforced via PSScriptAnalyzer settings).

> Testing tools are defined in the PowerShell unit test policy; do not redefine them here.

---

## 2. PowerShell Design & Safety

- Prefer **advanced functions** with `CmdletBinding()` and named parameters over ad-hoc script blocks.
- Add `[Parameter(Mandatory = $true)]` and validation attributes where appropriate; avoid positional parameters for user-facing scripts.
- For any state-changing action, implement **ShouldProcess/SupportsShouldProcess** and gate destructive behavior with `$PSCmdlet.ShouldProcess(...)`.
- Avoid global state and mutable script-scoped variables unless required; pass data explicitly.
- Avoid `Invoke-Expression`, plaintext secrets, and hard-coded credentials/paths. Use secure defaults.
- Use `Write-Error`/`throw` for failures; avoid silent catch-alls. Bubble errors unless you can add actionable context.

---

## 3. Structure, Naming, and Comments

- Keep scripts **cohesive and under 500 lines** (matches general policy).
- Use approved verbs and descriptive nouns for functions (PSScriptAnalyzer will enforce).
- Prefer modules/helpers over copy-paste between scripts; share common logic in dedicated helper functions.
- Comment **why** for non-obvious patterns (e.g., rule suppressions, compatibility shims), not what.

---

## 4. Running the Toolchain (PowerShell)

When PowerShell code changes, your toolchain loop must include:

1. `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."`
2. `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."`
3. (Type checking is not applicable for PowerShell; skip to testing.)
4. Run Pester per the unit test policy.

Rerun the loop from step 1 if any step changes code or fails.

> Install prerequisites once with `pwsh -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Install-PoshQCTools"` (installs PSScriptAnalyzer + Pester to CurrentUser).

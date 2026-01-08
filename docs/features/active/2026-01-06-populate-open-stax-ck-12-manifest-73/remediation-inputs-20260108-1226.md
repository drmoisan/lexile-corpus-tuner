# Remediation Inputs: Fix PowerShell Compliance

## 1. Required Fixes

### 1.1 `scripts/dev-tools/download-ck12-bundle.ps1` - Write-Host Removal
- **Location:** Lines 9, 14, 15, 18, 21, 22, 25, 28, 29.
- **Finding:** Script uses `Write-Host` for status messages.
- **Requirement:** Replace `Write-Host` with `Write-Output` (standard pipeline output) or `Write-Verbose` (for debug info). Since this is a dev script that outputs data for human inspection, `Write-Host` usage for headers/status is often replaced by `Write-Output` or strictly formatted output strings.
- **Acceptance Criteria:** `Invoke-PoshQCAnalyze` returns 0 warnings/errors.

### 1.2 `scripts/dev-tools/download-ck12-bundle.ps1` - Formatting
- **Location:** Line 6.
- **Finding:** Assignment statement not aligned.
- **Requirement:** Run the repo formatter `Invoke-PoshQCFormat`.
- **Acceptance Criteria:** `Invoke-PoshQCAnalyze` returns 0 warnings/errors.

## 2. Verification Commands

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command "Invoke-ScriptAnalyzer -Path ./scripts/dev-tools/download-ck12-bundle.ps1 -Settings ./scripts/powershell/PoshQC/settings/pssa.settings.psd1"
```

## 3. Constraints

- Do not change the Python code (it is already compliant).
- Do not disable PSScriptAnalyzer rules.

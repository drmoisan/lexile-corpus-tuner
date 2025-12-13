# Test Fixes Validation Guide

## What Was Fixed

The following test files were updated to be compatible with Pester v5:

1. **Pester v4 to v5 Migration** (19 total replacements):
   - `tests/powershell/PoshQC.Comprehensive.Tests.ps1` (11 instances)
   - `tests/scripts/dev-tools/fix-all.Tests.ps1` (2 instances)
   - `tests/scripts/dev-tools/link-parent-child.Tests.ps1` (3 instances)
   - `tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1` (3 instances)

2. **Import-ScriptFunction Standardization** (3 files):
   - `tests/powershell/dev-tools/run-cloc.Tests.ps1`
   - `tests/powershell/new-potential-entry.Tests.ps1`
   - `tests/scripts/dev-tools/tree.Tests.ps1`

## How to Validate

### 1. Run the Full Test Suite

```powershell
# From the repository root
pwsh -NoProfile -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."
```

### 2. Run Individual Test Files

```powershell
# Test the originally failing files mentioned in the request
Invoke-Pester -Path "tests/powershell/dev-tools/collect-commit-context.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/powershell/dev-tools.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/powershell/PoshQC.Comprehensive.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/scripts/dev-tools/run-actionlint.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/powershell/dev-tools/run-cloc.Tests.ps1" -Output Detailed
```

### 3. Run Additional Fixed Files

```powershell
# Test the additional files that were also fixed
Invoke-Pester -Path "tests/scripts/dev-tools/fix-all.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/scripts/dev-tools/link-parent-child.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/powershell/new-potential-entry.Tests.ps1" -Output Detailed
Invoke-Pester -Path "tests/scripts/dev-tools/tree.Tests.ps1" -Output Detailed
```

### 4. Run the Complete Toolchain

Per the repository's `general-code-change.instructions.md` policy, the full toolchain should be run:

```powershell
# 1. Format
pwsh -NoProfile -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."

# 2. Analyze (Lint)
pwsh -NoProfile -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."

# 3. Type Check
# (Not applicable for PowerShell)

# 4. Test
pwsh -NoProfile -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."
```

## Expected Results

All tests should pass with:
- 0 failed tests
- All assertions using Pester v5 syntax (`Should -Invoke` instead of `Assert-MockCalled`)
- All test files using the shared `TestHelpers.ps1` for `Import-ScriptFunction`

## Changes Summary

### Syntax Migration
- **Before (Pester v4):**
  ```powershell
  Assert-MockCalled -CommandName MyCommand -Times 1 -Exactly
  ```

- **After (Pester v5):**
  ```powershell
  Should -Invoke -CommandName MyCommand -Times 1 -Exactly
  ```

### Import Pattern Standardization
- **Before (Custom implementation):**
  ```powershell
  function global:Import-ScriptFunction {
      # ... custom implementation using [scriptblock]::Create() ...
  }
  ```

- **After (Shared helper):**
  ```powershell
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
  . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "path/to/TestHelpers.ps1"))
  ```

## Verification Checklist

- [ ] All 5 originally failing test files pass
- [ ] All 5 additionally fixed test files pass
- [ ] No `Assert-MockCalled` syntax remains in any test file
- [ ] No duplicate `Import-ScriptFunction` implementations using `[scriptblock]::Create()`
- [ ] Format step completes without changes
- [ ] Analyze step reports no errors
- [ ] Full test suite passes

## Files Modified

1. `tests/powershell/dev-tools/run-cloc.Tests.ps1`
2. `tests/powershell/PoshQC.Comprehensive.Tests.ps1`
3. `tests/scripts/dev-tools/fix-all.Tests.ps1`
4. `tests/scripts/dev-tools/link-parent-child.Tests.ps1`
5. `tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1`
6. `tests/powershell/new-potential-entry.Tests.ps1`
7. `tests/scripts/dev-tools/tree.Tests.ps1`

## Files Analyzed (No Changes Required)

1. `tests/scripts/dev-tools/run-actionlint.Tests.ps1` - Already using Pester v5 syntax
2. `tests/powershell/dev-tools.Tests.ps1` - No issues found
3. `tests/powershell/dev-tools/collect-commit-context.Tests.ps1` - No issues found

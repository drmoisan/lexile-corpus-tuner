# Test Fixes Summary

## Overview
Fixed failing unit tests in PowerShell test files to be compatible with Pester v5. The main issues were:
1. Duplicate `Import-ScriptFunction` implementations using incorrect methods
2. Outdated Pester v4 `Assert-MockCalled` syntax instead of Pester v5 `Should -Invoke`

## Files Modified

### 1. tests/scripts/dev-tools/run-cloc.Tests.ps1
**Problem:** Had its own duplicate implementation of `Import-ScriptFunction` that used `[scriptblock]::Create()` instead of using the shared `TestHelpers.ps1` implementation.

**Fix:** Replaced the duplicate function definition with a proper import of the shared TestHelpers.ps1 file.

**Changes:**
- Removed lines 3-34 (duplicate `Import-ScriptFunction` definition)
- Added lines 3-4:
  ```powershell
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
  . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "..\powershell\Support\TestHelpers.ps1"))
  ```

### 2. tests/scripts/powershell/PoshQC/PoshQC.Comprehensive.Tests.ps1
**Problem:** Used Pester v4 syntax (`Assert-MockCalled`) which is not compatible with Pester v5+.

**Fix:** Replaced all 11 instances of `Assert-MockCalled` with the Pester v5 equivalent `Should -Invoke`.

**Changes:** Updated mock assertion syntax on 11 lines throughout the file.

### 3. tests/scripts/dev-tools/fix-all.Tests.ps1
**Problem:** Used Pester v4 syntax (`Assert-MockCalled`).

**Fix:** Replaced 2 instances of `Assert-MockCalled` with `Should -Invoke`.

### 4. tests/scripts/dev-tools/link-parent-child.Tests.ps1
**Problem:** Used Pester v4 syntax (`Assert-MockCalled`).

**Fix:** Replaced 3 instances of `Assert-MockCalled` with `Should -Invoke`.

### 5. tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1
**Problem:** Used Pester v4 syntax (`Assert-MockCalled`).

**Fix:** Replaced 3 instances of `Assert-MockCalled` with `Should -Invoke`.

### 6. tests/scripts/dev-tools/new-potential-entry.Tests.ps1
**Problem:** Had a fallback `Import-ScriptFunction` implementation using `[scriptblock]::Create()`.

**Fix:** Removed the fallback implementation since TestHelpers.ps1 is already imported.

**Changes:**
- Removed lines 5-34 (conditional fallback function definition)
- TestHelpers.ps1 import on line 3 is sufficient

### 7. tests/scripts/dev-tools/tree.Tests.ps1
**Problem:** Had its own `Import-ScriptFunction` implementation using `[scriptblock]::Create()` in the BeforeAll block.

**Fix:** Added TestHelpers.ps1 import and removed the custom function.

**Changes:**
- Added line 3: `. (Join-Path $PSScriptRoot '..\powershell\Support\TestHelpers.ps1' -Resolve)`
- Removed the custom `Import-ScriptFunction` from BeforeAll block (lines 4-34)

## Verification Status

### All test files fixed:
1. ✅ tests/scripts/dev-tools/run-cloc.Tests.ps1
2. ✅ tests/scripts/powershell/PoshQC/PoshQC.Comprehensive.Tests.ps1
3. ✅ tests/scripts/dev-tools/fix-all.Tests.ps1
4. ✅ tests/scripts/dev-tools/link-parent-child.Tests.ps1
5. ✅ tests/scripts/dev-tools/sync-agents-from-instructions.Tests.ps1
6. ✅ tests/scripts/dev-tools/new-potential-entry.Tests.ps1
7. ✅ tests/scripts/dev-tools/tree.Tests.ps1

### Tests that were already compatible:
1. ✅ tests/scripts/dev-tools/run-actionlint.Tests.ps1 (already using Pester v5 syntax)
2. ✅ tests/powershell/dev-tools.Tests.ps1 (no issues)
3. ✅ tests/scripts/dev-tools/collect-commit-context.Tests.ps1 (no issues)

## Technical Details

### Pester v4 to v5 Migration
The key change in Pester v5 is the assertion syntax for mocks:

**Pester v4:**
```powershell
Assert-MockCalled -CommandName MyCommand -Times 1 -Exactly
```

**Pester v5:**
```powershell
Should -Invoke -CommandName MyCommand -Times 1 -Exactly
```

### TestHelpers.ps1 Import Pattern
The correct pattern for importing test helpers is:
```powershell
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
. (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "relative/path/to/TestHelpers.ps1"))
```

This ensures the helper functions are available in the test file's scope.

## Next Steps

To verify the fixes:
```powershell
# Run all tests
pwsh -NoProfile -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."

# Run specific test files
Invoke-Pester -Path "tests/scripts/dev-tools/run-cloc.Tests.ps1"
Invoke-Pester -Path "tests/scripts/powershell/PoshQC/PoshQC.Comprehensive.Tests.ps1"
Invoke-Pester -Path "tests/scripts/dev-tools/run-actionlint.Tests.ps1"
Invoke-Pester -Path "tests/powershell/dev-tools.Tests.ps1"
Invoke-Pester -Path "tests/scripts/dev-tools/collect-commit-context.Tests.ps1"
```

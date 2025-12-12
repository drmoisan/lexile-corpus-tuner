# TODO: Refactor fix-all.Tests.ps1 to Use Shared Test Helpers

**Status:** Backlog  
**Priority:** Low  
**Created:** 2025-12-11  
**Related PR:** Add comprehensive unit tests for fix-all.ps1

## Background

During the merge of feature/PoshQc-#21, a shared `tests/powershell/Support/TestHelpers.ps1` file was introduced that contains the `Import-ScriptFunction` helper used by multiple test files.

Currently, `tests/scripts/dev-tools/fix-all.Tests.ps1` still defines `Import-ScriptFunction` locally in its `BeforeAll` block instead of using the shared helper.

## Proposed Change

Refactor `tests/scripts/dev-tools/fix-all.Tests.ps1` to use the shared `TestHelpers.ps1` file, consistent with other test files like `tests/powershell/dev-tools/collect-commit-context.Tests.ps1`.

### Current Pattern (fix-all.Tests.ps1)
```powershell
BeforeAll {
    function global:Import-ScriptFunction {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Name
        )
        # ... implementation ...
    }
}
```

### Target Pattern (collect-commit-context.Tests.ps1)
```powershell
BeforeAll {
    $script:scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
    $script:helperPath = Join-Path -Path $script:scriptRoot -ChildPath "..\Support\TestHelpers.ps1"
    . (Resolve-Path -Path $script:helperPath)
    # ... rest of setup ...
}
```

## Benefits

- **Consistency**: Aligns with established pattern used by other test files
- **Maintainability**: Single source of truth for test helpers
- **DRY**: Eliminates code duplication across test files

## Implementation Notes

1. Update `BeforeAll` block to source the shared `TestHelpers.ps1`
2. Remove local `Import-ScriptFunction` definition
3. Adjust path calculations to account for different directory structure:
   - fix-all.Tests.ps1 is in `tests/scripts/dev-tools/`
   - TestHelpers.ps1 is in `tests/powershell/Support/`
   - Path would be: `..\..\powershell\Support\TestHelpers.ps1`
4. Run full test suite to verify no regressions
5. Run toolchain (format → lint → test) to ensure compliance

## Testing

- [ ] All 17 fix-all.Tests.ps1 tests continue to pass
- [ ] Tests pass on both PowerShell 5.1 and 7.x
- [ ] No linting or formatting issues introduced

## Acceptance Criteria

- `tests/scripts/dev-tools/fix-all.Tests.ps1` sources `TestHelpers.ps1` instead of defining helpers locally
- All tests pass
- Code follows established patterns from other test files
- No policy violations introduced

---

**Note:** This is a technical debt item. The current implementation works correctly (17/17 tests passing), so this refactoring can be deferred to a future sprint when time allows.

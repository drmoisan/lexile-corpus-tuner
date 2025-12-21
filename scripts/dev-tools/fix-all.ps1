#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Thin wrapper that delegates fix-all to the Python implementation.

.DESCRIPTION
    Runs Black, Ruff, Pyright, and Pytest (with coverage by default) via the
    Python module scripts.dev_tools.fix_all. Existing callers can continue to
    invoke this PowerShell script; all logic now lives in Python.

.PARAMETER MaxRuffRetries
    Maximum number of times to retry Ruff --fix before failing (default: 3)

.PARAMETER NoCoverage
    Skip coverage flags when invoking pytest.
#>

param(
    [int]$MaxRuffRetries = 3,
    [switch]$NoCoverage
)

function Invoke-FixAll {
    [CmdletBinding()]
    [OutputType([int])]
    param(
        [int]$MaxRuffRetries = 3,
        [switch]$NoCoverage
    )

    $pythonArgs = @(
        'run',
        'python',
        '-m',
        'scripts.dev_tools.fix_all',
        '--max-ruff-retries',
        $MaxRuffRetries
    )

    $disableCoverage = $NoCoverage.IsPresent
    if ($env:FIX_ALL_DISABLE_COVERAGE) {
        $disableCoverage = $disableCoverage -or ($env:FIX_ALL_DISABLE_COVERAGE -match '^(?i:true|1)$')
    }
    if ($disableCoverage) {
        $pythonArgs += '--no-coverage'
    }

    & poetry @pythonArgs
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    return $exitCode
}

if ($MyInvocation.InvocationName -eq '.') {
    return
}

if ($env:POSHQC_SKIP_SCRIPT_EXECUTION) {
    return
}

$exitCode = Invoke-FixAll -MaxRuffRetries $MaxRuffRetries -NoCoverage:$NoCoverage
exit $exitCode

#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Runs all code quality checks with auto-fix and intelligent retry logic.

.DESCRIPTION
    This script runs Black, Ruff, Pyright, and Pytest in sequence with the following logic:
    1. Run Black (auto-fix formatting)
    2. Run Ruff with --fix (auto-fix linting), retry if needed
    3. Re-run Black and Ruff to ensure consistency
    4. Run Pyright (halt on failure)
    5. Run Pytest with coverage (halt on failure)
    6. Confirm all checks pass if successful

.PARAMETER MaxRuffRetries
    Maximum number of times to retry Ruff fix before halting (default: 3)
#>

param(
    [int]$MaxRuffRetries = 3
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Output ""
    Write-Output "==> $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Output "[OK] $Message"
}

function Write-Failure {
    param([string]$Message)
    Write-Error $Message
}

function Invoke-Command-WithStatus {
    param(
        [string[]]$CommandParts,
        [string]$StepName
    )

    Write-Step $StepName
    if (-not $CommandParts -or $CommandParts.Count -eq 0) {
        throw "CommandParts cannot be empty."
    }

    $command = $CommandParts[0]
    $commandArguments = @()
    if ($CommandParts.Count -gt 1) {
        $commandArguments = $CommandParts[1..($CommandParts.Count - 1)]
    }

    $output = & $command @commandArguments 2>&1
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }

    if ($output) {
        $output | Out-Host
    }

    return $exitCode
}

# Step 1: Run Black formatting
Write-Step "Step 1: Running Black formatting..."
$exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'black', '.') -StepName "Black: format"
if ($exitCode -ne 0) {
    Write-Failure "Black formatting failed. Please review errors above."
    exit 1
}
Write-Success "Black formatting completed successfully"

# Step 2: Run Ruff with fix, retry if needed
Write-Step "Step 2: Running Ruff linting with auto-fix..."
$ruffAttempt = 0
$ruffSuccess = $false

while ($ruffAttempt -lt $MaxRuffRetries) {
    $ruffAttempt++
    Write-Output "Ruff attempt $ruffAttempt of $MaxRuffRetries..."

    $exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'ruff', 'check', '--fix') -StepName "Ruff: fix"

    if ($exitCode -eq 0) {
        $ruffSuccess = $true
        Write-Success "Ruff linting passed"
        break
    }

    if ($ruffAttempt -lt $MaxRuffRetries) {
        Write-Output "Ruff found issues. Retrying..."
    }
}

if (-not $ruffSuccess) {
    Write-Failure "Ruff linting failed after $MaxRuffRetries attempts. Please review errors above."
    exit 1
}

# Step 3: Re-run Black and Ruff to ensure consistency
Write-Step "Step 3: Re-running Black to ensure consistency..."
$exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'black', '.') -StepName "Black: format (verify)"
if ($exitCode -ne 0) {
    Write-Failure "Black formatting failed on verification pass."
    exit 1
}
Write-Success "Black formatting verified"

Write-Step "Step 4: Re-running Ruff to verify fixes..."
$exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'ruff', 'check') -StepName "Ruff: lint (verify)"
if ($exitCode -ne 0) {
    Write-Failure "Ruff linting still has issues after fixes. Please review errors above."
    exit 1
}
Write-Success "Ruff linting verified"

# Step 5: Run Pyright type checking
Write-Step "Step 5: Running Pyright type checking..."
$exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'pyright') -StepName "Pyright: type-check"
if ($exitCode -ne 0) {
    Write-Failure "Pyright type checking failed. Please review errors above."
    exit 1
}
Write-Success "Pyright type checking passed"

# Step 6: Run Pytest with coverage
Write-Step "Step 6: Running Pytest with coverage..."
$exitCode = Invoke-Command-WithStatus -CommandParts @('poetry', 'run', 'pytest', '--cov=src/lexile_corpus_tuner', '--cov-report=term-missing') -StepName "Pytest: test with coverage"
if ($exitCode -ne 0) {
    Write-Failure "Pytest failed. Please review errors above."
    exit 1
}
Write-Success "Pytest passed"

# All checks passed
Write-Output ""
Write-Output "========================================"
Write-Output "ALL CHECKS PASSED"
Write-Output "========================================"
Write-Output "  Black formatting: PASS"
Write-Output "  Ruff linting: PASS"
Write-Output "  Pyright type checking: PASS"
Write-Output "  Pytest with coverage: PASS"
Write-Output "========================================"

exit 0




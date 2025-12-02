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
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Invoke-Command-WithStatus {
    param(
        [string]$Command,
        [string]$StepName
    )
    
    Write-Step $StepName
    $output = Invoke-Expression $Command 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($output) {
        Write-Host $output
    }
    
    return $exitCode
}

# Step 1: Run Black formatting
Write-Step "Step 1: Running Black formatting..."
$exitCode = Invoke-Command-WithStatus "poetry run black ." "Black: format"
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
    Write-Host "Ruff attempt $ruffAttempt of $MaxRuffRetries..." -ForegroundColor Yellow
    
    $exitCode = Invoke-Command-WithStatus "poetry run ruff check --fix" "Ruff: fix"
    
    if ($exitCode -eq 0) {
        $ruffSuccess = $true
        Write-Success "Ruff linting passed"
        break
    }
    
    if ($ruffAttempt -lt $MaxRuffRetries) {
        Write-Host "Ruff found issues. Retrying..." -ForegroundColor Yellow
    }
}

if (-not $ruffSuccess) {
    Write-Failure "Ruff linting failed after $MaxRuffRetries attempts. Please review errors above."
    exit 1
}

# Step 3: Re-run Black and Ruff to ensure consistency
Write-Step "Step 3: Re-running Black to ensure consistency..."
$exitCode = Invoke-Command-WithStatus "poetry run black ." "Black: format (verify)"
if ($exitCode -ne 0) {
    Write-Failure "Black formatting failed on verification pass."
    exit 1
}
Write-Success "Black formatting verified"

Write-Step "Step 4: Re-running Ruff to verify fixes..."
$exitCode = Invoke-Command-WithStatus "poetry run ruff check" "Ruff: lint (verify)"
if ($exitCode -ne 0) {
    Write-Failure "Ruff linting still has issues after fixes. Please review errors above."
    exit 1
}
Write-Success "Ruff linting verified"

# Step 5: Run Pyright type checking
Write-Step "Step 5: Running Pyright type checking..."
$exitCode = Invoke-Command-WithStatus "poetry run pyright" "Pyright: type-check"
if ($exitCode -ne 0) {
    Write-Failure "Pyright type checking failed. Please review errors above."
    exit 1
}
Write-Success "Pyright type checking passed"

# Step 6: Run Pytest with coverage
Write-Step "Step 6: Running Pytest with coverage..."
$exitCode = Invoke-Command-WithStatus "poetry run pytest --cov=src/lexile_corpus_tuner --cov-report=term-missing" "Pytest: test with coverage"
if ($exitCode -ne 0) {
    Write-Failure "Pytest failed. Please review errors above."
    exit 1
}
Write-Success "Pytest passed"

# All checks passed
Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  • Black formatting: " -NoNewline
Write-Host "PASS" -ForegroundColor Green
Write-Host "  • Ruff linting: " -NoNewline
Write-Host "PASS" -ForegroundColor Green
Write-Host "  • Pyright type checking: " -NoNewline
Write-Host "PASS" -ForegroundColor Green
Write-Host "  • Pytest with coverage: " -NoNewline
Write-Host "PASS" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

exit 0

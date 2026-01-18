#!/usr/bin/env pwsh
<#
.SYNOPSIS
Runs PowerShell linting (PSScriptAnalyzer) for the repo.

.DESCRIPTION
This is a thin entrypoint wrapper used by repo tooling/docs.
It imports the local PoshQC module and runs Invoke-PoshQCAnalyze across the repo root.

.EXITCODES
0 on success; non-zero on failure.
#>

$ErrorActionPreference = "Stop"

try {
    Import-Module ./scripts/powershell/PoshQC -Force
    Invoke-PoshQCAnalyze -Root .
}
catch {
    Write-Error $_
    exit 1
}

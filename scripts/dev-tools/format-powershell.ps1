#!/usr/bin/env pwsh
<#[
.SYNOPSIS
Formats PowerShell files in the repo.

.DESCRIPTION
This is a thin entrypoint wrapper used by repo tooling/docs.
It imports the local PoshQC module and runs Invoke-PoshQCFormat across the repo root.

.EXITCODES
0 on success; non-zero on failure.
#>

$ErrorActionPreference = "Stop"

try {
    Import-Module ./scripts/powershell/PoshQC -Force
    Invoke-PoshQCFormat -Root .
}
catch {
    Write-Error $_
    exit 1
}

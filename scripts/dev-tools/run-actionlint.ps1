#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run actionlint, downloading a local copy into tools/actionlint/bin if needed.

Place this file at:
  scripts/dev-tools-/run-actionlint.ps1

actionlint will be stored at:
  tools/actionlint/bin/actionlint.exe
#>

$ErrorActionPreference = 'Stop'
$InformationPreference = 'Continue'

# Resolve repo root relative to this script:
# scripts/dev-tools-/run-actionlint.ps1 → ../.. = repo root
$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $scriptDir '..\..')
$binDir = Join-Path $repoRoot 'tools\actionlint\bin'
$exePath = Join-Path $binDir 'actionlint.exe'

# Try to find actionlint on PATH first
$actionlintCmd = $null
try {
    $cmd = Get-Command actionlint -ErrorAction Stop
    $actionlintCmd = $cmd.Source
} catch {
    Write-Information 'actionlint not found on PATH; will try local copy under tools/actionlint/bin.' -InformationAction Continue
}

# Fall back to tools/actionlint/bin if it's already present
if (-not $actionlintCmd -and (Test-Path $exePath)) {
    $actionlintCmd = $exePath
}

if (-not $actionlintCmd) {
    Write-Information 'actionlint not found; downloading local copy into tools/actionlint/bin...' -InformationAction Continue

    New-Item -ItemType Directory -Force -Path $binDir | Out-Null

    # Pin a specific version so CI and agents get deterministic behavior
    $ACTIONLINT_VERSION = '1.7.7'

    $zipName = "actionlint_${ACTIONLINT_VERSION}_windows_amd64.zip"
    $zipPath = Join-Path $binDir $zipName
    $zipUrl = "https://github.com/rhysd/actionlint/releases/download/v$ACTIONLINT_VERSION/$zipName"

    Write-Information "Downloading $zipUrl ..." -InformationAction Continue
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

    Write-Information "Extracting $zipName to $binDir ..." -InformationAction Continue
    Expand-Archive -Path $zipPath -DestinationPath $binDir -Force
    Remove-Item $zipPath -Force

    if (-not (Test-Path $exePath)) {
        throw "Downloaded actionlint archive, but actionlint.exe was not found in $binDir"
    }

    $actionlintCmd = $exePath
}

# Ensure tools/actionlint/bin is on PATH for this process so child tools can find it
if (-not ($env:PATH.Split([IO.Path]::PathSeparator) -contains $binDir)) {
    $env:PATH = "$binDir$([IO.Path]::PathSeparator)$env:PATH"
}

Write-Information 'Running actionlint...' -InformationAction Continue

# Pass through all arguments to actionlint
& $actionlintCmd @args
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Error "actionlint exited with code $exitCode"
    exit $exitCode
}

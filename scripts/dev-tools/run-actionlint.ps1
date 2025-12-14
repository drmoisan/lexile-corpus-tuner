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

function Resolve-ActionlintPath {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath
    )

    $scriptDir = Split-Path -Parent $ScriptPath
    $repoRoot = (Resolve-Path (Join-Path $scriptDir '..\..')).Path
    $binDir = Join-Path $repoRoot 'tools\actionlint\bin'
    $exePath = Join-Path $binDir 'actionlint.exe'

    return [PSCustomObject]@{
        RepoRoot = $repoRoot
        BinDir   = $binDir
        ExePath  = $exePath
    }
}

function Find-ActionlintOnPath {
    [CmdletBinding()]
    param()

    try {
        $cmd = Get-Command actionlint -ErrorAction Stop
        return $cmd.Source
    } catch {
        Write-Information 'actionlint not found on PATH; will try local copy under tools/actionlint/bin.' -InformationAction Continue
        return $null
    }
}

function Install-Actionlint {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$BinDir,

        [Parameter(Mandatory = $true)]
        [string]$ExePath,

        [Parameter(Mandatory = $false)]
        [string]$Version = '1.7.7'
    )

    Write-Information 'actionlint not found; downloading local copy into tools/actionlint/bin...' -InformationAction Continue

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

    $zipName = "actionlint_${Version}_windows_amd64.zip"
    $zipPath = Join-Path $BinDir $zipName
    $zipUrl = "https://github.com/rhysd/actionlint/releases/download/v$Version/$zipName"

    Write-Information "Downloading $zipUrl ..." -InformationAction Continue
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

    Write-Information "Extracting $zipName to $BinDir ..." -InformationAction Continue
    Expand-Archive -Path $zipPath -DestinationPath $BinDir -Force
    Remove-Item $zipPath -Force

    if (-not (Test-Path $ExePath)) {
        throw "Downloaded actionlint archive, but actionlint.exe was not found in $BinDir"
    }

    return $ExePath
}

function Add-DirectoryToPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory
    )

    if (-not ($env:PATH.Split([IO.Path]::PathSeparator) -contains $Directory)) {
        $env:PATH = "$Directory$([IO.Path]::PathSeparator)$env:PATH"
    }
}

function Invoke-ActionlintCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandPath,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    Write-Information 'Running actionlint...' -InformationAction Continue

    & $CommandPath @Arguments
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Error "actionlint exited with code $exitCode"
        exit $exitCode
    }
}

# Main script execution
# Flow: Resolve-ActionlintPath -> Find-ActionlintOnPath -> Install-Actionlint -> Add-DirectoryToPath -> Invoke-ActionlintCommand
$paths = Resolve-ActionlintPath -ScriptPath $PSCommandPath

# Try to find actionlint on PATH first
$actionlintCmd = Find-ActionlintOnPath

# Fall back to tools/actionlint/bin if it's already present
if (-not $actionlintCmd -and (Test-Path $paths.ExePath)) {
    $actionlintCmd = $paths.ExePath
}

# Download if not found
if (-not $actionlintCmd) {
    $actionlintCmd = Install-Actionlint -BinDir $paths.BinDir -ExePath $paths.ExePath
}

# Ensure tools/actionlint/bin is on PATH for this process so child tools can find it
Add-DirectoryToPath -Directory $paths.BinDir

# Pass through all arguments to actionlint
Invoke-ActionlintCommand -CommandPath $actionlintCmd -Arguments $args






[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepoRoot,
    [string]$InputPath,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'

$resolvedRepoRoot = if ($RepoRoot) {
    (Resolve-Path -Path $RepoRoot -ErrorAction Stop).Path
} else {
    $parentPath = Join-Path -Path $PSScriptRoot -ChildPath '..'
    $grandParentPath = Join-Path -Path $parentPath -ChildPath '..'
    $repoRootCandidate = Join-Path -Path $grandParentPath -ChildPath '..'
    (Resolve-Path -Path $repoRootCandidate -ErrorAction Stop).Path
}

$resolvedInputPath = if ($InputPath) {
    $InputPath
} else {
    Join-Path -Path $resolvedRepoRoot -ChildPath 'artifacts/pester/powershell-coverage.xml'
}

$resolvedOutputPath = if ($OutputPath) {
    $OutputPath
} else {
    Join-Path -Path $resolvedRepoRoot -ChildPath 'artifacts/pester/powershell-coverage.koverage.xml'
}

Import-Module (Join-Path $PSScriptRoot 'PoshQC.psm1') -Force

if ($PSCmdlet.ShouldProcess($resolvedOutputPath, 'Convert coverage paths to relative paths')) {
    Convert-PoshQCCoverageToRelative -InputPath $resolvedInputPath -OutputPath $resolvedOutputPath -RepoRoot $resolvedRepoRoot
}

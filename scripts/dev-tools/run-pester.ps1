[CmdletBinding()]
param(
    [Parameter()]
    [string] $Root = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Delegate to the repo's PowerShell quality tooling module so test settings stay centralized.
$poshQcModulePath = Join-Path -Path $PSScriptRoot -ChildPath "..\powershell\PoshQC\PoshQC.psd1"
Import-Module $poshQcModulePath -Force

Invoke-PoshQCTest -Root $Root

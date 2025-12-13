#!/usr/bin/env pwsh
# Quick test script to validate the test fixes

Write-Host "Testing Pester availability..." -ForegroundColor Cyan
$pester = Get-Module -ListAvailable -Name Pester | Select-Object -First 1
if ($pester) {
    Write-Host "Pester version: $($pester.Version)" -ForegroundColor Green
} else {
    Write-Host "Pester not found!" -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning PoshQC.Comprehensive.Tests.ps1..." -ForegroundColor Cyan
$result = Invoke-Pester -Path "tests\powershell\PoshQC.Comprehensive.Tests.ps1" -PassThru

Write-Host "`nTest Results:" -ForegroundColor Cyan
Write-Host "  Passed: $($result.PassedCount)" -ForegroundColor Green
Write-Host "  Failed: $($result.FailedCount)" -ForegroundColor $(if ($result.FailedCount -gt 0) { "Red" } else { "Green" })
Write-Host "  Skipped: $($result.SkippedCount)" -ForegroundColor Yellow

if ($result.FailedCount -gt 0) {
    Write-Host "`nFailed tests:" -ForegroundColor Red
    $result.Failed | ForEach-Object {
        Write-Host "  - $($_.ExpandedName)" -ForegroundColor Red
    }
    exit 1
}

Write-Host "`nAll tests passed!" -ForegroundColor Green
exit 0

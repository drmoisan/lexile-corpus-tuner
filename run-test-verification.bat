@echo off
echo Running test verification...
echo.

pwsh.exe -NoProfile -NoLogo -NonInteractive -Command ^
"try { ^
    Import-Module ./scripts/powershell/PoshQC -ErrorAction Stop; ^
    Write-Host 'PoshQC module loaded successfully' -ForegroundColor Green; ^
    Write-Host ''; ^
    Write-Host 'Running tests...' -ForegroundColor Cyan; ^
    $result = Invoke-PoshQCTest -Root . -PassThru; ^
    Write-Host ''; ^
    Write-Host 'Test Results:' -ForegroundColor Cyan; ^
    Write-Host '  Passed:  ' $result.PassedCount -ForegroundColor Green; ^
    Write-Host '  Failed:  ' $result.FailedCount -ForegroundColor $(if ($result.FailedCount -gt 0) {'Red'} else {'Green'}); ^
    Write-Host '  Skipped: ' $result.SkippedCount -ForegroundColor Yellow; ^
    exit $result.FailedCount ^
} catch { ^
    Write-Error $_.Exception.Message; ^
    exit 1 ^
}"

echo.
if %ERRORLEVEL% EQU 0 (
    echo All tests passed!
) else (
    echo Tests failed with exit code %ERRORLEVEL%
)

Set-StrictMode -Version Latest

BeforeAll {
    $env:POSHQC_SKIP_SCRIPT_EXECUTION = '1'
    $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\fix-all.ps1"
    . $script:scriptPath
}

Describe "fix-all.ps1 helpers" {
    Context "Write-Step" {
        It "emits step messages with prefix" {
            $output = & { Write-Step "Doing work" }
            $output | Should -Contain "==> Doing work"
        }

        It "handles empty message" {
            $output = & { Write-Step "" }
            ($output -join "`n") | Should -Match "==>"
        }

        It "handles special characters in message" {
            $output = & { Write-Step "Test & <special> 'chars'" }
            $output | Should -Contain "==> Test & <special> 'chars'"
        }
    }

    Context "Write-Success" {
        It "emits success messages with prefix" {
            $output = & { Write-Success "All good" }
            $output | Should -Contain "[OK] All good"
        }

        It "handles empty message" {
            $output = & { Write-Success "" }
            ($output -join "") | Should -Be "[OK] "
        }

        It "handles special characters in message" {
            $output = & { Write-Success "Test & <special> 'chars'" }
            $output | Should -Contain "[OK] Test & <special> 'chars'"
        }
    }

    Context "Write-Failure" {
        BeforeEach {
            $script:lastError = $null
            Mock -CommandName Write-Error -MockWith { param($Message) $script:lastError = $Message }
        }

        It "writes failures via Write-Error" {
            Write-Failure "bad news"
            $script:lastError | Should -Be "bad news"
        }

        It "handles empty message" {
            Write-Failure ""
            $script:lastError | Should -Be ""
        }

        It "handles special characters in message" {
            Write-Failure "Error & <special> 'chars'"
            $script:lastError | Should -Be "Error & <special> 'chars'"
        }
    }

    Context "Invoke-Command-WithStatus" {
        BeforeEach {
            . $script:scriptPath
            Mock -CommandName Write-Step -MockWith { }
        }

        It "runs command with single argument and returns exit code 0" {
            $global:LASTEXITCODE = 0
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "demo") -StepName "demo-step"
            $result | Should -Be 0
            Assert-MockCalled -CommandName Write-Step -Times 1
        }

        It "returns exit code when command fails" {
            $global:LASTEXITCODE = 1
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "fail") -StepName "demo-step"
            $result | Should -Be 1
        }

        It "throws when command parts are empty" {
            { Invoke-Command-WithStatus -CommandParts @() -StepName "demo-step" } | Should -Throw
        }

        It "passes multiple arguments to command" {
            $global:LASTEXITCODE = 0
            Mock -CommandName Write-Output -MockWith { param($first, $second) "${first}-${second}" }
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "hello", "world") -StepName "demo-step"
            $result | Should -Be 0
        }
    }
}

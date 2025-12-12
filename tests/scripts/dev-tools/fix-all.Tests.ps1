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

Describe "Invoke-FixAll" {
    BeforeEach {
        $script:steps = New-Object System.Collections.Generic.List[string]
        $script:lastFailure = $null
        Mock -CommandName Write-Step -MockWith { param($Message) $script:steps.Add($Message) | Out-Null }
        Mock -CommandName Write-Success -MockWith { }
        Mock -CommandName Write-Failure -MockWith { param($Message) $script:lastFailure = $Message }
        Mock -CommandName Write-Output -MockWith { param($Message) $null = $Message }
    }

    It "returns success when all commands succeed" {
        Mock -CommandName Invoke-Command-WithStatus -MockWith {
            param($CommandParts, $StepName)
            $null = $CommandParts
            $null = $StepName
            return 0
        }

        $result = Invoke-FixAll -MaxRuffRetries 1
        $result | Should -Be 0
        Assert-MockCalled -CommandName Invoke-Command-WithStatus -Times 6
        $script:steps | Should -Contain "Step 6: Running Pytest with coverage..."
    }

    It "retries Ruff and returns failure when retries exhausted" {
        $script:attempts = 0
        Mock -CommandName Invoke-Command-WithStatus -MockWith {
            param($CommandParts, $StepName)
            $null = $CommandParts
            if ($StepName -like "Ruff: fix") {
                $script:attempts++ | Out-Null
                return 1
            }
            return 0
        }

        $result = Invoke-FixAll -MaxRuffRetries 2
        $result | Should -Be 1
        $script:lastFailure | Should -Match "Ruff linting failed"
        $script:attempts | Should -Be 2
    }

    It "returns failure when verification passes fail" {
        Mock -CommandName Invoke-Command-WithStatus -MockWith {
            param($CommandParts, $StepName)
            $null = $CommandParts
            switch ($StepName) {
                "Ruff: lint (verify)" { return 1 }
                default { return 0 }
            }
        }

        $result = Invoke-FixAll -MaxRuffRetries 1
        $result | Should -Be 1
        $script:lastFailure | Should -Match "Ruff linting still has issues"
    }

    It "returns failure when initial Black pass fails" {
        Mock -CommandName Invoke-Command-WithStatus -MockWith {
            param($CommandParts, $StepName)
            $null = $CommandParts
            if ($StepName -eq "Black: format") { return 1 }
            return 0
        }

        $result = Invoke-FixAll -MaxRuffRetries 1
        $result | Should -Be 1
        $script:lastFailure | Should -Match "Black formatting failed"
    }

    It "returns failure when Pytest fails" {
        Mock -CommandName Invoke-Command-WithStatus -MockWith {
            param($CommandParts, $StepName)
            $null = $CommandParts
            if ($StepName -eq "Pytest: test with coverage") { return 1 }
            return 0
        }

        $result = Invoke-FixAll -MaxRuffRetries 1
        $result | Should -Be 1
        $script:lastFailure | Should -Match "Pytest failed"
    }
}

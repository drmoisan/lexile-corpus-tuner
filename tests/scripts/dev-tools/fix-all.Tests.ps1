Set-StrictMode -Version Latest

BeforeAll {
    function global:Import-ScriptFunction {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Name
        )

        $resolved = (Resolve-Path -Path $Path).Path
        if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
            $null = $null
            $errors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile($resolved, [ref]$null, [ref]$errors)
            if ($errors -and $errors.Count -gt 0) {
                throw "Failed to parse ${resolved}: $($errors[0].Message)"
            }

            $funcAst = $ast.Find(
                {
                    param($node)
                    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq $Name
                },
                $true
            )

            if (-not $funcAst) {
                throw "Function $Name not found in $resolved"
            }

            return [scriptblock]::Create($funcAst.Extent.Text)
        }
    }
}

Describe "fix-all.ps1 helpers" {
    Context "Write-Step" {
        BeforeEach {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\fix-all.ps1"
            . (Import-ScriptFunction -Path $scriptPath -Name "Write-Step")
        }

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
        BeforeEach {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\fix-all.ps1"
            . (Import-ScriptFunction -Path $scriptPath -Name "Write-Success")
        }

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
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\fix-all.ps1"
            . (Import-ScriptFunction -Path $scriptPath -Name "Write-Failure")
        }

        It "writes failures via Write-Error" {
            $script:lastError = $null
            Mock -CommandName Write-Error -MockWith { param($Message) $script:lastError = $Message }
            Write-Failure "bad news"
            $script:lastError | Should -Be "bad news"
        }

        It "handles empty message" {
            $script:lastError = $null
            Mock -CommandName Write-Error -MockWith { param($Message) $script:lastError = $Message }
            Write-Failure ""
            $script:lastError | Should -Be ""
        }

        It "handles special characters in message" {
            $script:lastError = $null
            Mock -CommandName Write-Error -MockWith { param($Message) $script:lastError = $Message }
            Write-Failure "Error & <special> 'chars'"
            $script:lastError | Should -Be "Error & <special> 'chars'"
        }
    }

    Context "Invoke-Command-WithStatus" {
        BeforeEach {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\fix-all.ps1"
            . (Import-ScriptFunction -Path $scriptPath -Name "Write-Step")
            . (Import-ScriptFunction -Path $scriptPath -Name "Invoke-Command-WithStatus")
            Mock -CommandName Write-Step -MockWith { }
        }

        It "runs command with single argument and returns exit code 0" {
            $global:LASTEXITCODE = 0
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "demo") -StepName "demo-step"
            $result | Should -Be 0
        }

        It "runs command with multiple arguments" {
            $global:LASTEXITCODE = 0
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "arg1", "arg2", "arg3") -StepName "multi-arg"
            $result | Should -Be 0
        }

        It "returns non-zero exit code when command fails" {
            $global:LASTEXITCODE = 1
            $mockCommand = {
                param($arg1)
                $null = $arg1
                $global:LASTEXITCODE = 1
            }
            Mock -CommandName Test-Path -MockWith $mockCommand
            $result = Invoke-Command-WithStatus -CommandParts @("Test-Path", "nonexistent") -StepName "fail-step"
            $result | Should -Be 1
        }

        It "returns 0 when LASTEXITCODE is null" {
            $global:LASTEXITCODE = $null
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "test") -StepName "null-exit"
            $result | Should -Be 0
        }

        It "throws when CommandParts is empty array" {
            { Invoke-Command-WithStatus -CommandParts @() -StepName "empty-parts" } | Should -Throw "CommandParts cannot be empty."
        }

        It "throws when CommandParts is null" {
            { Invoke-Command-WithStatus -CommandParts $null -StepName "null-parts" } | Should -Throw "CommandParts cannot be empty."
        }

        It "captures and outputs command output" {
            Mock -CommandName Out-Host -MockWith { }
            $global:LASTEXITCODE = 0
            $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "test output") -StepName "output-test"
            $result | Should -Be 0
            Assert-MockCalled -CommandName Out-Host -Times 1
        }

        It "handles command with no output" {
            $mockCommand = {
                $global:LASTEXITCODE = 0
            }
            Mock -CommandName Get-Date -MockWith $mockCommand
            $result = Invoke-Command-WithStatus -CommandParts @("Get-Date") -StepName "no-output"
            $result | Should -Be 0
        }
    }
}

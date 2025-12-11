Set-StrictMode -Version Latest

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

Describe "link-parent-child.ps1 - Read-IssueNumber" {
    It "trims provided issue number" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        $result = Read-IssueNumber -Label "child" -Value " 42 "
        $result | Should -Be "42"
    }

    It "errors when no issue number supplied" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-ScriptError")
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Read-Host -MockWith { "" }

        { Read-IssueNumber -Label "parent" -Value "" } | Should -Not -Throw
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "required"
    }

    It "prompts user when issue number is empty" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        Mock -CommandName Read-Host -MockWith { "123" }

        $result = Read-IssueNumber -Label "child" -Value ""
        $result | Should -Be "123"
        Assert-MockCalled -CommandName Read-Host -Times 1
    }

    It "prompts user when issue number is whitespace" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        Mock -CommandName Read-Host -MockWith { "456" }

        $result = Read-IssueNumber -Label "parent" -Value "   "
        $result | Should -Be "456"
    }
}

Describe "link-parent-child.ps1 - Test-GhCli" {
    It "succeeds when gh is available" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Test-GhCli")
        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "gh" } -MockWith {
            [pscustomobject]@{ Name = "gh"; Source = "/usr/bin/gh" }
        }

        { Test-GhCli } | Should -Not -Throw
    }

    It "errors when gh is not found" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Test-GhCli")
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-ScriptError")
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "gh" } -MockWith { $null }

        Test-GhCli
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "gh CLI not found"
    }
}

Describe "link-parent-child.ps1 - Get-Issue" {
    It "returns parsed JSON when gh succeeds" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-Issue")
        $mockJson = '{"number":42,"title":"Test Issue","url":"https://github.com/test/repo/issues/42","body":"Issue body"}'
        Mock -CommandName gh -MockWith {
            $global:LASTEXITCODE = 0
            return $mockJson
        }

        $result = Get-Issue -IssueNumber "42" -Label "test"
        $result | Should -Not -BeNullOrEmpty
        $result.number | Should -Be 42
        $result.title | Should -Be "Test Issue"
        $result.url | Should -Be "https://github.com/test/repo/issues/42"
        $result.body | Should -Be "Issue body"
    }

    It "errors when gh command fails" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-Issue")
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-ScriptError")
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName gh -MockWith {
            $global:LASTEXITCODE = 1
            return ""
        }

        Get-Issue -IssueNumber "999" -Label "parent"
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "Unable to fetch parent issue"
    }

    It "errors when gh returns empty output" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-Issue")
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-ScriptError")
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName gh -MockWith {
            $global:LASTEXITCODE = 0
            return ""
        }

        Get-Issue -IssueNumber "555" -Label "child"
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "Unable to fetch child issue"
    }
}

Set-StrictMode -Version Latest

BeforeAll {
    $env:POSHQC_SKIP_SCRIPT_EXECUTION = '1'
    $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-parent-child.ps1"
    . $script:scriptPath
}

Describe "link-parent-child.ps1 - Read-IssueNumber" {
    It "trims provided issue number" {
        $result = Read-IssueNumber -Label "child" -Value " 42 "
        $result | Should -Be "42"
    }

    It "errors when no issue number supplied" {
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Read-Host -MockWith { "" }

        { Read-IssueNumber -Label "parent" -Value "" } | Should -Not -Throw
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "required"
    }

    It "prompts user when issue number is empty" {
        Mock -CommandName Read-Host -MockWith { "123" }

        $result = Read-IssueNumber -Label "child" -Value ""
        $result | Should -Be "123"
        Should -Invoke -CommandName Read-Host -Times 1
    }

    It "prompts user when issue number is whitespace" {
        Mock -CommandName Read-Host -MockWith { "456" }

        $result = Read-IssueNumber -Label "parent" -Value "   "
        $result | Should -Be "456"
    }
}

Describe "link-parent-child.ps1 - Test-GhCli" {
    It "succeeds when gh is available" {
        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "gh" } -MockWith {
            [pscustomobject]@{ Name = "gh"; Source = "/usr/bin/gh" }
        }

        { Test-GhCli } | Should -Not -Throw
    }

    It "errors when gh is not found" {
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
        $mockJson = '{"number":42,"title":"Test Issue","url":"https://github.com/test/repo/issues/42","body":"Issue body"}'
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            [void] $GhArgs
            $global:LASTEXITCODE = 0
            return @{ Output = $mockJson; ExitCode = 0 }
        }

        $result = Get-Issue -IssueNumber "42" -Label "test"
        $result | Should -Not -BeNullOrEmpty
        $result.number | Should -Be 42
        $result.title | Should -Be "Test Issue"
        $result.url | Should -Be "https://github.com/test/repo/issues/42"
        $result.body | Should -Be "Issue body"
    }

    It "errors when gh command fails" {
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            [void] $GhArgs
            $global:LASTEXITCODE = 1
            return @{ Output = ""; ExitCode = 1 }
        }

        Get-Issue -IssueNumber "999" -Label "parent"
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "Unable to fetch parent issue"
    }

    It "errors when gh returns empty output" {
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            [void] $GhArgs
            $global:LASTEXITCODE = 0
            return @{ Output = ""; ExitCode = 0 }
        }

        Get-Issue -IssueNumber "555" -Label "child"
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "Unable to fetch child issue"
    }
}

Describe "link-parent-child.ps1 - Invoke-LinkParentChild" {
    BeforeEach {
        $script:messages = @()
        $script:ghCalls = @()
        $script:ghExitByVerb = @{}
        Mock -CommandName Test-GhCli -MockWith { }
        Mock -CommandName Write-Output -MockWith {
            param([Parameter(ValueFromRemainingArguments = $true)]$Message, $InputObject)
            if ($PSBoundParameters.ContainsKey('InputObject')) {
                $script:messages += $InputObject
            } else {
                $script:messages += $Message
            }
        }
        Mock -CommandName Set-Content -MockWith { param($Path, $Value, $Encoding) $script:lastWrite = @{ Path = $Path; Value = $Value; Encoding = $Encoding } }
        Mock -CommandName Remove-Item -MockWith { }
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            [void] $GhArgs
            $script:ghCalls += , $GhArgs
            $operation = if ($GhArgs -contains 'edit') { 'edit' } elseif ($GhArgs -contains 'comment') { 'comment' } else { $GhArgs[1] }
            if ($script:ghExitByVerb.ContainsKey($operation)) {
                $global:LASTEXITCODE = $script:ghExitByVerb[$operation]
            } else {
                $global:LASTEXITCODE = 0
            }

            return @{ Output = ""; ExitCode = $global:LASTEXITCODE }
        }
    }

    It "updates parent body and comments on child when not already linked" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'child body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = "## Child Issues`n- [ ] #2 - Existing`n" } }
        Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10'

        $script:lastWrite.Value | Should -Match "Child Issues"
        $script:lastWrite.Value | Should -Match "#1"
        $script:messages | Should -Contain "Updated parent issue #10 with child link."
        $script:messages | Should -Contain "Added parent link comment to child issue #1."
    }

    It "skips updates when parent already lists child and child links back" {
        $parentBody = "## Child Issues`n- [ ] #1 - Child title"
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'Contains #10' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = $parentBody } }

        Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10'

        Should -Invoke -CommandName Set-Content -Times 0
        $script:ghCalls.Count | Should -Be 0
        $script:messages | Should -Contain "No parent body changes were required."
        $script:messages | Should -Contain "Child issue already references parent #10; no comment added."
    }

    It "adds child section when missing and user agrees" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = "Intro" } }
        Mock -CommandName Read-Host -MockWith { 'y' }
        Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10'

        $script:lastWrite.Value | Should -Match "## Child Issues"
        $script:lastWrite.Value | Should -Match "#1"
        Should -Invoke -CommandName Read-Host -Times 1
        $script:messages | Should -Contain "Updated parent issue #10 with child link."
        $script:messages | Should -Contain "Added parent link comment to child issue #1."
    }

    It "throws when parent body is empty" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child'; url = 'https://example.com/1'; body = 'child body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent'; url = 'https://example.com/10'; body = '   ' } }

        $action = { Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10' }
        Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Parent issue #10 has an empty body*'
    }

    It "throws when gh edit fails" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'child body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = "## Child Issues`n" } }
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            [void] $GhArgs
            $global:LASTEXITCODE = 1
            return @{ Output = ""; ExitCode = 1 }
        }

        $action = { Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10' }
        Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Failed to update parent issue #10.'
    }

    It "throws when adding comment fails" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'child body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = "## Child Issues`n" } }
        Mock -CommandName Invoke-GhCli -MockWith {
            param([string[]]$GhArgs)
            $script:ghCalls += , $GhArgs
            if ($GhArgs -contains 'comment') {
                $global:LASTEXITCODE = 1
            } else {
                $global:LASTEXITCODE = 0
            }

            return @{ Output = ""; ExitCode = $global:LASTEXITCODE }
        }

        $threw = $false
        $errorMessage = $null
        try {
            Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10'
        } catch {
            $threw = $true
            $errorMessage = $_.Exception.Message
        }

        ($script:ghCalls | Where-Object { $_ -contains 'comment' }).Count | Should -BeGreaterThan 0
        $threw | Should -BeTrue
        $errorMessage | Should -Match 'Failed to add parent link comment to child issue #1.'
    }

    It "throws when user declines adding child section" {
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '1' } -MockWith { [pscustomobject]@{ number = 1; title = 'Child title'; url = 'https://example.com/1'; body = 'body' } }
        Mock -CommandName Get-Issue -ParameterFilter { $IssueNumber -eq '10' } -MockWith { [pscustomobject]@{ number = 10; title = 'Parent title'; url = 'https://example.com/10'; body = "Intro" } }
        Mock -CommandName Read-Host -MockWith { 'n' }

        $action = { Invoke-LinkParentChild -ChildIssueNumberParam '1' -ParentIssueNumberParam '10' }
        Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage "Aborting: parent issue lacks a 'Child Issues' section*declined."
    }
}


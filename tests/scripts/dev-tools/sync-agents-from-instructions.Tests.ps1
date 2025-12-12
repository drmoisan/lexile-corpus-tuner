Set-StrictMode -Version Latest

Describe "sync-agents-from-instructions.ps1" {
    BeforeAll {
        $env:POSHQC_SKIP_SCRIPT_EXECUTION = '1'
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\sync-agents-from-instructions.ps1"
        . $script:scriptPath
    }

    Context "Get-InstructionsBody" {
        It "throws when file is missing" {
            Mock -CommandName Test-Path -MockWith { $false }

            { Get-InstructionsBody -Path "missing.md" } | Should -Throw -ExpectedMessage "Instructions file not found: missing.md"
        }

        It "returns trimmed content without frontmatter" {
            $content = @"
---
applyTo: "**"
---

line1
line2
"@
            Mock -CommandName Test-Path -MockWith { $true }
            Mock -CommandName Get-Content -MockWith { $content }

            $result = Get-InstructionsBody -Path "dummy.md"
            $result | Should -Be "line1`nline2"
        }
    }

    Context "Get-AgentContent" {
        BeforeEach {
            $script:copilotPath = Join-Path -Path "/repo" -ChildPath ".github/copilot-instructions.md"
            $script:instructionsDir = Join-Path -Path "/repo" -ChildPath ".github/instructions"
            Mock -CommandName Get-InstructionsBody -MockWith {
                param($Path)
                switch ($Path) {
                    { $_ -eq $script:copilotPath } { return "copilot body" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "general-code-change.instructions.md") } { return "general code" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "general-unit-test.instructions.md") } { return "general unit" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "github-actions.instructions.md") } { return "gh actions" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "python-code-change.instructions.md") } { return "python code" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "python-unit-test.instructions.md") } { return "python unit" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "powershell-code-change.instructions.md") } { return "ps code" }
                    { $_ -eq (Join-Path -Path $script:instructionsDir -ChildPath "powershell-unit-test.instructions.md") } { return "ps unit" }
                    default { throw "Unexpected path $Path" }
                }
            }
        }

        It "builds AGENTS content with all sections" {
            $result = Get-AgentContent -RepoRootParam "/repo"

            $result.Path | Should -Be "/repo/AGENTS.md"
            $result.Content | Should -Match "# AGENTS.md"
            $result.Content | Should -Match "copilot body"
            $result.Content | Should -Match "general code"
            $result.Content | Should -Match "ps unit"
        }
    }

    Context "Invoke-SyncAgentInstruction" {
        It "writes generated content to AGENTS.md" {
            $expected = [pscustomobject]@{ Path = "/work/AGENTS.md"; Content = "content" }
            Mock -CommandName Get-AgentContent -MockWith { param($RepoRootParam) [void]$RepoRootParam; $expected }
            Mock -CommandName Set-Content -MockWith { }
            Mock -CommandName Write-Output -MockWith { }

            Invoke-SyncAgentInstruction -RepoRootParam "/work"

            Assert-MockCalled -CommandName Get-AgentContent -Times 1 -ParameterFilter { $RepoRootParam -eq "/work" }
            Assert-MockCalled -CommandName Set-Content -Times 1 -ParameterFilter { $LiteralPath -eq "/work/AGENTS.md" -and $Value -eq "content" -and $NoNewline }
            Assert-MockCalled -CommandName Write-Output -Times 1
        }
    }
}

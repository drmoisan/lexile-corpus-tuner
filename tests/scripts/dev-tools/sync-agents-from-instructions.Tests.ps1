Set-StrictMode -Version Latest

BeforeAll {
    function global:Import-ScriptFunction {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Name
        )

        $resolved = (Resolve-Path -Path $Path).Path
        if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
            $errors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile($resolved, [ref]$null, [ref]$errors)
            if ($errors -and $errors.Count -gt 0) {
                throw "Failed to parse ${resolved}: $($errors[0].Message)"
            }

            $funcAst = $ast.Find(
                {
                    param($node)
                    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $Name
                },
                $true
            )

            if (-not $funcAst) {
                throw "Function $Name not found in $resolved"
            }

            return [scriptblock]::Create($funcAst.Extent.Text)
        }
    }

    $script:SyncAgentsScriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\sync-agents-from-instructions.ps1"
}

Describe "sync-agents-from-instructions.ps1" {
    Context "Get-InstructionsBody" {
        BeforeAll {
            $script:GetInstructionsBody = Import-ScriptFunction -Path $SyncAgentsScriptPath -Name "Get-InstructionsBody"
        }

        It "strips YAML frontmatter and trims body content" {
            # Arrange
            $instructionsPath = "C:/repo/.github/instructions/general-code-change.instructions.md"
            $rawContent = @"
---
applyTo: "**"
name: general-code-change-policy
---

  Body content that should remain.
"@

            Mock -CommandName Test-Path -MockWith { $true } -ParameterFilter { $LiteralPath -eq $instructionsPath }
            Mock -CommandName Get-Content -MockWith { $rawContent } -ParameterFilter { $LiteralPath -eq $instructionsPath }

            # Act
            $result = & $GetInstructionsBody -Path $instructionsPath

            # Assert
            $result | Should -Be "Body content that should remain."
        }

        It "throws when the instructions file is missing" {
            # Arrange
            $missingPath = "C:/repo/.github/instructions/missing.instructions.md"
            Mock -CommandName Test-Path -MockWith { $false } -ParameterFilter { $LiteralPath -eq $missingPath }

            # Act / Assert
            { & $GetInstructionsBody -Path $missingPath } | Should -Throw "Instructions file not found: $missingPath"
        }
    }

    Context "Script execution" {
        It "writes AGENTS.md with header and ordered instruction sections" {
            # Arrange
            $repoRoot = "C:/repo"
            $copilotPath = Join-Path $repoRoot ".github/copilot-instructions.md"
            $agentsPath = Join-Path $repoRoot "AGENTS.md"

            $generalCodePath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/general-code-change.instructions.md"
            $generalTestPath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/general-unit-test.instructions.md"
            $githubActionsPath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/github-actions.instructions.md"
            $pythonCodePath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/python-code-change.instructions.md"
            $pythonTestPath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/python-unit-test.instructions.md"
            $powershellCodePath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/powershell-code-change.instructions.md"
            $powershellTestPath = Join-Path -Path $repoRoot -ChildPath ".github/instructions/powershell-unit-test.instructions.md"

            $instructionBodies = @{}
            $instructionBodies[$copilotPath] = @"
---
applyTo: "**"
---

Copilot guidance
"@
            $instructionBodies[$generalCodePath] = "General code guidance"
            $instructionBodies[$generalTestPath] = "Unit test guidance"
            $instructionBodies[$githubActionsPath] = "Workflow guidance"
            $instructionBodies[$pythonCodePath] = "Python code guidance"
            $instructionBodies[$pythonTestPath] = "Python unit test guidance"
            $instructionBodies[$powershellCodePath] = "PowerShell code guidance"
            $instructionBodies[$powershellTestPath] = "PowerShell unit test guidance"

            Mock -CommandName Test-Path -MockWith { param($LiteralPath) $instructionBodies.ContainsKey($LiteralPath) }
            Mock -CommandName Get-Content -MockWith { param($LiteralPath) $instructionBodies[$LiteralPath] }

            $script:CapturedContent = $null
            Mock -CommandName Set-Content -MockWith {
                param($LiteralPath, $Value, $NoNewline)
                $script:CapturedContent = [PSCustomObject]@{ LiteralPath = $LiteralPath; Value = $Value; NoNewline = $NoNewline }
            }

            Mock -CommandName Write-Output -MockWith { }

            # Act
            & $SyncAgentsScriptPath -RepoRoot $repoRoot

            # Assert
            Assert-MockCalled Set-Content -Times 1 -ParameterFilter { $LiteralPath -eq $agentsPath -and $NoNewline }

            $expectedContent = @"
# AGENTS.md

> NOTE: This file is **generated** from:
>
> - .github/copilot-instructions.md
> - .github/instructions/general-code-change.instructions.md
> - .github/instructions/general-unit-test.instructions.md
> - .github/instructions/github-actions.instructions.md
> - .github/instructions/python-code-change.instructions.md
> - .github/instructions/python-unit-test.instructions.md
> - .github/instructions/powershell-code-change.instructions.md
> - .github/instructions/powershell-unit-test.instructions.md
>
> Do not edit this file manually.
> To update policies, edit the source *.instructions.md files and
> `.github/copilot-instructions.md`, then run:
>
>   pwsh -File scripts/dev-tools/sync-agents-from-instructions.ps1

## Repository Setup (High-Level)

- For coding and testing policies, always follow the sections below in the order:
  Copilot instructions → general policies → language-specific policies → CI policies.
- Use the language- and domain-specific sections for Python, PowerShell, and CI behavior.


## Repository Instructions (GitHub Copilot Canonical)

<!-- BEGIN: copilot-instructions -->
Copilot guidance

<!-- END: copilot-instructions -->

## General Code Change Policy

<!-- BEGIN: general-code-change -->
General code guidance

<!-- END: general-code-change -->

## General Unit Test Policy

<!-- BEGIN: general-unit-test -->
Unit test guidance

<!-- END: general-unit-test -->

## GitHub Actions Workflow Policy

<!-- BEGIN: github-actions -->
Workflow guidance

<!-- END: github-actions -->

## Python Code Change Policy

<!-- BEGIN: python-code-change -->
Python code guidance

<!-- END: python-code-change -->

## Python Unit Test Policy

<!-- BEGIN: python-unit-test -->
Python unit test guidance

<!-- END: python-unit-test -->

## PowerShell Code Change Policy

<!-- BEGIN: powershell-code-change -->
PowerShell code guidance

<!-- END: powershell-code-change -->

## PowerShell Unit Test Policy

<!-- BEGIN: powershell-unit-test -->
PowerShell unit test guidance

<!-- END: powershell-unit-test -->
"@

            $CapturedContent.LiteralPath | Should -Be $agentsPath
            $CapturedContent.NoNewline | Should -BeTrue
            $CapturedContent.Value | Should -Be $expectedContent.TrimEnd()
            $CapturedContent.Value | Should -Not -Match "^---"
        }

        It "throws when any instructions file cannot be found" {
            # Arrange
            $repoRoot = "C:/repo"
            $missingPath = Join-Path $repoRoot ".github/instructions/general-unit-test.instructions.md"
            Mock -CommandName Test-Path -MockWith { param($LiteralPath) $LiteralPath -ne $missingPath }
            Mock -CommandName Get-Content -MockWith { "placeholder" }

            # Act / Assert
            { & $SyncAgentsScriptPath -RepoRoot $repoRoot } | Should -Throw "Instructions file not found: $missingPath"
        }
    }
}





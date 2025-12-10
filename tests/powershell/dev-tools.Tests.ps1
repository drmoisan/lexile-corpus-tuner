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

Describe "collect-commit-context.ps1" {
    BeforeEach {
        $script:ReportOutput = "mock-output.txt"
        $script:captured = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Add-Content -MockWith {
            param($Path, $Value)
            $null = $Path
            $script:captured.Add($Value)
        }
    }

    It "writes section headers and content" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-commit-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Add-ReportSection")
        Add-ReportSection -Title "Test Section" -Cmd { "line1`nline2" }

        $script:captured.Count | Should -Be 2
        $script:captured[0] | Should -Match "===== Test Section ====="
        $script:captured[1] | Should -Match "line1`nline2"
    }

    It "writes placeholder when allowed to fail" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-commit-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Add-ReportSection")
        Add-ReportSection -Title "MayFail" -Cmd { throw "boom" } -AllowFail
        $script:captured[1] | Should -Match "\[n/a\]"
    }
}

Describe "collect-pull-request-context.ps1 helpers" {
    It "formats brace rename paths" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-pull-request-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Format-DiffPath")
        $result = Format-DiffPath -PathText "dir/{old => new}/file.txt"
        $result | Should -Be "dir/new/file.txt"
    }

    It "converts numstat text to totals and file list" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-pull-request-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "ConvertFrom-Numstat")
        $num = @"
4	2	file1.ps1
1	0	dir/file2.psm1
"@
        $result = ConvertFrom-Numstat -NumstatText $num
        $result.Additions | Should -Be 5
        $result.Deletions | Should -Be 2
        $result.Files | Should -Contain "file1.ps1"
        $result.Files | Should -Contain "dir/file2.psm1"
    }

    It "summarizes extensions with counts" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-pull-request-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Format-DiffPath")
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-ExtensionSummary")
        $summary = Get-ExtensionSummary -Files @("a.ps1", "b.ps1", "c")
        $summary | Should -Match "\s2\s+\.ps1"
        $summary | Should -Match "\s1\s+\(noext\)"
    }

    It "collects unique issue references" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-pull-request-context.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-IssueReference")
        $refs = Get-IssueReference -Text "Fixes #12 and relates to ABC-99 plus #12 again"
        $refs | Should -Be @("#12", "ABC-99")
    }
}

Describe "fix-all.ps1 helpers" {
    It "emits step messages with prefix" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\fix-all.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-Step")
        $output = & { Write-Step "Doing work" }
        $output | Should -Contain "==> Doing work"
    }

    It "emits success messages with prefix" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\fix-all.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-Success")
        $output = & { Write-Success "All good" }
        $output | Should -Contain "[OK] All good"
    }

    It "writes failures via Write-Error" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\fix-all.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-Failure")
        $script:lastError = $null
        Mock -CommandName Write-Error -MockWith { param($Message) $script:lastError = $Message }
        Write-Failure "bad news"
        $script:lastError | Should -Be "bad news"
    }

    It "runs command parts and returns exit code" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\fix-all.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-Step")
        . (Import-ScriptFunction -Path $scriptPath -Name "Invoke-Command-WithStatus")

        Mock -CommandName Write-Step -MockWith { }

        $global:LASTEXITCODE = 0
        $result = Invoke-Command-WithStatus -CommandParts @("Write-Output", "demo") -StepName "demo-step"
        $result | Should -Be 0
    }
}

Describe "link-feature-docs.ps1 helpers" {
    It "replaces existing section content" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\link-feature-docs.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Set-OrAppendSection")
        $content = @"
## Intro
hello

## Feature Docs
old body
"@
        $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew body"
        $updated | Should -Match "new body"
        $updated | Should -Not -Match "old body"
    }

    It "appends new section when missing" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\link-feature-docs.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Set-OrAppendSection")
        $content = "## Intro`nhello"
        $replacement = "## Feature Docs`ncontent"
        $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement $replacement
        $pattern = ([regex]::Escape($replacement.TrimEnd())) -replace "\\n", "\r?\n"
        $updated | Should -Match $pattern
    }
}

Describe "link-parent-child.ps1 helpers" {
    It "trims provided issue number" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        $result = Read-IssueNumber -Label "child" -Value " 42 "
        $result | Should -Be "42"
    }

    It "errors when no issue number supplied" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\link-parent-child.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Read-IssueNumber")
        . (Import-ScriptFunction -Path $scriptPath -Name "Write-ScriptError")
        $script:errors = New-Object System.Collections.Generic.List[string]
        Mock -CommandName Write-ScriptError -MockWith { param($Message) $script:errors.Add($Message) }
        Mock -CommandName Read-Host -MockWith { "" }

        { Read-IssueNumber -Label "parent" -Value "" } | Should -Not -Throw
        $script:errors.Count | Should -Be 1
        $script:errors[0] | Should -Match "required"
    }
}

Describe "new-active-feature-folder.ps1 helpers" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-active-feature-folder.ps1"
    }

    Context "Format-Checklist" {
        BeforeAll {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Format-Checklist")
        }

        It "normalizes checklist bullets" {
            $checklistInput = "Item one`n- existing`n"
            $result = Format-Checklist -Text $checklistInput
            $lines = $result -split "`r?`n" | ForEach-Object { $_.Trim() }
            $lines | Should -Contain "- [ ] Item one"
            $lines | Should -Contain "- existing"
        }

        It "converts plain text lines to unchecked items" {
            $testInput = "First item`nSecond item"
            $result = Format-Checklist -Text $testInput
            $result | Should -Match "- \[ \] First item"
            $result | Should -Match "- \[ \] Second item"
        }

        It "preserves existing checkbox format with brackets" {
            $testInput = "- [ ] Already formatted`n- [x] Completed"
            $result = Format-Checklist -Text $testInput
            $result | Should -Match "- \[ \] Already formatted"
            $result | Should -Match "- \[x\] Completed"
        }

        It "preserves lines starting with dash but no checkbox" {
            $testInput = "- Simple bullet`n- Another bullet"
            $result = Format-Checklist -Text $testInput
            $result | Should -Match "- Simple bullet"
            $result | Should -Match "- Another bullet"
        }

        It "removes empty lines" {
            $testInput = "Item one`n`n`nItem two"
            $result = Format-Checklist -Text $testInput
            ($result -split "`r?`n").Count | Should -Be 2
        }

        It "removes whitespace-only lines" {
            $testInput = "Item one`n   `nItem two"
            $result = Format-Checklist -Text $testInput
            ($result -split "`r?`n").Count | Should -Be 2
        }

        It "handles empty string input" {
            $result = Format-Checklist -Text ""
            $result | Should -Be ""
        }

        It "trims whitespace from each line" {
            $testInput = "  Item with spaces  `n  Another item  "
            $result = Format-Checklist -Text $testInput
            $result | Should -Match "- \[ \] Item with spaces"
            $result | Should -Match "- \[ \] Another item"
        }

        It "handles mixed line endings" {
            $testInput = "Item one`r`nItem two`nItem three"
            $result = Format-Checklist -Text $testInput
            ($result -split "`r?`n").Count | Should -Be 3
        }
    }

    Context "Get-Section" {
        BeforeAll {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-Section")
        }

        It "extracts named sections" {
            $content = "## Header`nline1`n## Next`nline2"
            (Get-Section -Content $content -Name "Header") | Should -Be "line1"
        }

        It "extracts multiline section content" {
            $content = "## Section`nline1`nline2`nline3`n## Next"
            $result = Get-Section -Content $content -Name "Section"
            $result | Should -Match "line1"
            $result | Should -Match "line2"
            $result | Should -Match "line3"
        }

        It "returns empty string when section not found" {
            $content = "## Other Section`ncontent"
            $result = Get-Section -Content $content -Name "Missing"
            $result | Should -Be ""
        }

        It "handles section at end of document" {
            $content = "## First`nfirst content`n## Last`nlast content"
            $result = Get-Section -Content $content -Name "Last"
            $result | Should -Be "last content"
        }

        It "handles section with special characters in name" {
            $content = "## Problem / Why`nThe problem is...`n## Next"
            $result = Get-Section -Content $content -Name "Problem / Why"
            $result | Should -Be "The problem is..."
        }

        It "trims leading and trailing whitespace from section content" {
            $content = "## Header`n  content with spaces  `n## Next"
            $result = Get-Section -Content $content -Name "Header"
            $result | Should -Be "content with spaces"
        }

        It "handles empty section" {
            $content = "## Empty`n## Next"
            $result = Get-Section -Content $content -Name "Empty"
            $result | Should -Be ""
        }

        It "handles section with varying whitespace around header" {
            $content = "##   Spaced Header  `ncontent`n## Next"
            $result = Get-Section -Content $content -Name "Spaced Header"
            $result | Should -Be "content"
        }
    }

    Context "Set-Section" {
        BeforeAll {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Set-Section")
        }

        It "sets or replaces sections" {
            $content = "## Header`nold`n"
            $updated = Set-Section -Content $content -Name "Header" -Body "new"
            $updated | Should -Match "## Header"
            $updated | Should -Match "new"
            $updated | Should -Not -Match "old"
        }

        It "replaces existing section content while preserving other sections" {
            $content = "## First`nfirst content`n## Second`nold second`n## Third`nthird content"
            $updated = Set-Section -Content $content -Name "Second" -Body "new second"
            $updated | Should -Match "first content"
            $updated | Should -Match "new second"
            $updated | Should -Not -Match "old second"
            $updated | Should -Match "third content"
        }

        It "appends new section when not present" {
            $content = "## Existing`nexisting content"
            $updated = Set-Section -Content $content -Name "New Section" -Body "new content"
            $updated | Should -Match "## Existing"
            $updated | Should -Match "## New Section"
            $updated | Should -Match "new content"
        }

        It "returns unchanged content when body is empty" {
            $content = "## Header`ncontent"
            $updated = Set-Section -Content $content -Name "Header" -Body ""
            $updated | Should -Be $content
        }

        It "returns unchanged content when body is whitespace only" {
            $content = "## Header`ncontent"
            $updated = Set-Section -Content $content -Name "Header" -Body "   "
            $updated | Should -Be $content
        }

        It "respects ShouldProcess when WhatIf is specified" {
            $content = "## Header`nold"
            $updated = Set-Section -Content $content -Name "Header" -Body "new" -WhatIf
            $updated | Should -Be $content
        }

        It "handles multiline body content" {
            $content = "## Header`nold"
            $body = "line1`nline2`nline3"
            $updated = Set-Section -Content $content -Name "Header" -Body $body
            $updated | Should -Match "line1"
            $updated | Should -Match "line2"
            $updated | Should -Match "line3"
        }

        It "handles section with special characters in name" {
            $content = "## Problem / Why`nold problem"
            $updated = Set-Section -Content $content -Name "Problem / Why" -Body "new problem"
            $updated | Should -Match "new problem"
            $updated | Should -Not -Match "old problem"
        }
    }

    Context "Set-HeaderPlaceholder" {
        BeforeAll {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Set-HeaderPlaceholder")
        }

        BeforeEach {
            $script:FeatureName = "example-feature"
            $script:issueField = "#123"
            $script:ownerField = "testowner"
            $script:updatedField = "2025-01-15"
        }

        It "replaces common placeholders" {
            $content = "- Owner: name`n- Last Updated: YYYY-MM-DD`n<feature-name> #<id>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Not -Match "<feature-name>"
            $result | Should -Not -Match "YYYY-MM-DD"
        }

        It "replaces feature-name placeholder" {
            $content = "Feature: <feature-name>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "Feature: example-feature"
        }

        It "replaces refactor-name placeholder" {
            $content = "Refactor: <refactor-name>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "Refactor: example-feature"
        }

        It "replaces epic-name placeholder" {
            $content = "Epic: <epic-name>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "Epic: example-feature"
        }

        It "replaces generic name placeholder" {
            $content = "Name: <name>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "Name: example-feature"
        }

        It "replaces issue id placeholders" {
            $content = 'Issue #<id> and #<id>'
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match 'Issue #123'
        }

        It "replaces tracking-issue placeholder" {
            $content = "Tracking: #<tracking-issue>"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match 'Tracking: #123'
        }

        It "replaces owner placeholder" {
            $content = "- Owner: name"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "- Owner: testowner"
        }

        It "replaces last updated placeholder" {
            $content = "- Last Updated: YYYY-MM-DD"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Match "- Last Updated: 2025-01-15"
        }

        It "replaces multiple occurrences of the same placeholder" {
            $content = "<feature-name> and <feature-name> again"
            $result = Set-HeaderPlaceholder -Content $content
            ($result -split "example-feature").Count | Should -Be 3
        }

        It "respects ShouldProcess when WhatIf is specified" {
            $content = "- Owner: name"
            $result = Set-HeaderPlaceholder -Content $content -WhatIf
            $result | Should -Be $content
        }

        It "handles content with no placeholders" {
            $content = "No placeholders here"
            $result = Set-HeaderPlaceholder -Content $content
            $result | Should -Be $content
        }

        It "handles empty content" {
            $result = Set-HeaderPlaceholder -Content ""
            $result | Should -Be ""
        }
    }
}

Describe "new-potential-entry.ps1 validation" {
    It "contains short name validation pattern" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-potential-entry.ps1"
        (Get-Content -Path $scriptPath -Raw) | Should -Match "\^\[a-z0-9\]\+\(-\[a-z0-9\]\+\)\*\$"
    }
}

Describe "potential-to-issue.ps1 helpers" {
    It "extracts a section by heading" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-Section")
        $script:content = "## Problem / Why`nabc`n## Proposed Behavior`ndef"
        $result = Get-Section -name "Problem / Why"
        $result | Should -Be "abc"
    }

    It "inserts or updates metadata lines" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Set-LineValue")
        $lines = New-Object System.Collections.Generic.List[string]
        $lines.AddRange([string[]]@("# Title", "- Issue: #1"))
        $metaEnd = 2
        $metaRef = [ref]$metaEnd

        Set-LineValue -arr $lines -label "Issue URL" -value "https://example" -metaEndRef $metaRef
        $lines | Should -Contain "- Issue URL: https://example"

        Set-LineValue -arr $lines -label "Issue" -value "#2" -metaEndRef $metaRef
        ($lines | Where-Object { $_ -like "- Issue:*" }) | Should -Contain "- Issue: #2"
    }
}

Describe "run-cloc.ps1" {
    It "throws when no bundled cloc is found" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\run-cloc.ps1"
        Mock -CommandName Resolve-Path -MockWith { param($Path) $null = $Path; [pscustomobject]@{ Path = "C:\repo" } }
        Mock -CommandName Test-Path -MockWith { $false }

        { & $scriptPath -Path "C:\repo" } | Should -Throw "Bundled cloc binary not found."
    }
}

Describe "tree.ps1" {
    It "lists entries while honoring exclusions and hidden flag" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\tree.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Show-Tree")
        $items = @(
            [pscustomobject]@{ Name = "visible.txt"; FullName = "C:\root\visible.txt"; PSIsContainer = $false; Attributes = [IO.FileAttributes]::Normal },
            [pscustomobject]@{ Name = ".git"; FullName = "C:\root\.git"; PSIsContainer = $true; Attributes = [IO.FileAttributes]::Directory },
            [pscustomobject]@{ Name = "folder"; FullName = "C:\root\folder"; PSIsContainer = $true; Attributes = [IO.FileAttributes]::Directory }
        )

        Mock -CommandName Get-ChildItem -MockWith {
            param($LiteralPath, $Force)
            $null = $Force
            if ($LiteralPath -eq "C:\root") {
                return $items
            }
            return @()
        }

        $output = Show-Tree -Path "C:\root" -ExcludeNames @(".git") -IncludeHiddenEntries -DirectoriesOnly:$false
        ($output -join "`n") | Should -Not -Match "\.git"
        ($output -join "`n") | Should -Match "visible.txt"
        ($output -join "`n") | Should -Match "folder"
    }
}

Describe "load-openai-key.ps1" {
    It "sets environment variable when secret is returned" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\src\lexile_corpus_tuner\lexile_scoring_model\pipeline_scripts\load-openai-key.ps1"
        $setCalls = New-Object System.Collections.Generic.List[hashtable]

        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "lpass" } -MockWith { [pscustomobject]@{ Name = "lpass" } }
        Mock -CommandName lpass -MockWith { $global:LASTEXITCODE = 0; "secret-value" }
        Mock -CommandName Set-Item -MockWith { param($Path, $Value) $setCalls.Add(@{ Path = $Path; Value = $Value }) }

        & $scriptPath -ItemName "Test Item" -EnvVar "TEST_ENV"

        $setCalls.Count | Should -Be 1
        $setCalls[0].Path | Should -Be "Env:TEST_ENV"
        $setCalls[0].Value | Should -Be "secret-value"
    }
}

Describe "Convert-PoshQCCoverageToRelative" {
    It "strips repo root prefix from coverage content" {
        $modulePath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\powershell\PoshQC\PoshQC.psm1"
        Import-Module -Name $modulePath -Force

        $repoRoot = 'D:\repos\lexile-corpus-tuner'
        $coverageContent = '<file path="D:\repos\lexile-corpus-tuner\scripts\demo.ps1" />'

        $result = Convert-PoshQCCoverageToRelative -InputContent $coverageContent -RepoRoot $repoRoot -PassThru

        $result | Should -Match 'scripts\\demo.ps1'
        $result | Should -Not -Match ([regex]::Escape($repoRoot))
    }

    It "strips repo root prefix when coverage uses forward slashes" {
        $modulePath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\powershell\PoshQC\PoshQC.psm1"
        Import-Module -Name $modulePath -Force

        $repoRoot = 'D:\repos\lexile-corpus-tuner'
        $coverageContent = '<file path="D:/repos/lexile-corpus-tuner/scripts/demo.ps1" />'

        $result = Convert-PoshQCCoverageToRelative -InputContent $coverageContent -RepoRoot $repoRoot -PassThru

        $result | Should -Match 'scripts/demo.ps1'
        $result | Should -Not -Match ([regex]::Escape($repoRoot))
    }
}

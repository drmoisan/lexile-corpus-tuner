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
    It "normalizes checklist bullets" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-active-feature-folder.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Format-Checklist")
        $checklistInput = "Item one`n- existing`n"
        $result = Format-Checklist -Text $checklistInput
        $lines = $result -split "`r?`n" | ForEach-Object { $_.Trim() }
        $lines | Should -Contain "- [ ] Item one"
        $lines | Should -Contain "- existing"
    }

    It "extracts named sections" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-active-feature-folder.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-Section")
        $content = "## Header`nline1`n## Next`nline2"
        (Get-Section -Content $content -Name "Header") | Should -Be "line1"
    }

    It "sets or replaces sections" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-active-feature-folder.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Set-Section")
        $content = "## Header`nold`n"
        $updated = Set-Section -Content $content -Name "Header" -Body "new"
        $updated | Should -Match "## Header"
        $updated | Should -Match "new"
        $updated | Should -Not -Match "old"
    }

    It "replaces common placeholders" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-active-feature-folder.ps1"
        . (Import-ScriptFunction -Path $scriptPath -Name "Set-HeaderPlaceholder")
        $script:FeatureName = "example_feature"
        $script:issueField = "#1"
        $script:ownerField = "owner"
        $script:updatedField = "2025-01-01"
        $content = "- Owner: name`n- Last Updated: YYYY-MM-DD`n<feature-name> #<id>"
        $result = Set-HeaderPlaceholder -Content $content
        $result | Should -Not -Match "<feature-name>"
        $result | Should -Not -Match "YYYY-MM-DD"
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

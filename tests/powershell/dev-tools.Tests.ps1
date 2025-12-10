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

Describe "collect-pull-request-context.ps1" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\collect-pull-request-context.ps1"
    }

    Context "Invoke-Git" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
        }

        It "executes git command successfully with string output" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "output" }
            $result = Invoke-Git -Args @('status')
            $result.Out | Should -Be "output"
            $result.Code | Should -Be 0
        }

        It "executes git command successfully with array output" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; @("line1", "line2") }
            $result = Invoke-Git -Args @('log')
            $result.Out | Should -Be "line1`nline2"
            $result.Code | Should -Be 0
        }

        It "handles null output" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; $null }
            $result = Invoke-Git -Args @('status')
            $result.Out | Should -Be ""
            $result.Code | Should -Be 0
        }

        It "throws on non-zero exit when AllowNonZeroExit is false" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 1; "error output" }
            { Invoke-Git -Args @('invalid') } | Should -Throw "git invalid failed*"
        }

        It "does not throw on non-zero exit when AllowNonZeroExit is true" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 1; "error output" }
            $result = Invoke-Git -Args @('invalid') -AllowNonZeroExit
            $result.Out | Should -Be "error output"
            $result.Code | Should -Be 1
        }
    }

    Context "Write-Section" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
        }

        It "formats section title with borders" {
            $result = Write-Section -Title "Test Section"
            $result | Should -Match "===== Test Section ====="
            $result | Should -Match "^\n"
            $result | Should -Match "\n$"
        }

        It "handles empty title" {
            $result = Write-Section -Title ""
            $result | Should -Match "====="
        }
    }

    Context "Get-ItemCount" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ItemCount")
        }

        It "returns 0 for null input" {
            $result = Get-ItemCount -x $null
            $result | Should -Be 0
        }

        It "returns 1 for single item" {
            $result = Get-ItemCount -x "single"
            $result | Should -Be 1
        }

        It "returns count for array" {
            $result = Get-ItemCount -x @("a", "b", "c")
            $result | Should -Be 3
        }

        It "returns 1 for empty array" {
            $result = Get-ItemCount -x @()
            $result | Should -Be 0
        }
    }

    Context "Select-DefaultBase" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Select-DefaultBase")
        }

        It "returns first valid ref from candidate list" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3)
                $null = $Command, $Args1, $Args2
                $global:LASTEXITCODE = if ($Args3 -eq 'origin/main') { 0 } else { 1 }
                if ($global:LASTEXITCODE -eq 0) { "abc123" } else { "" }
            }
            $result = Select-DefaultBase
            $result | Should -Be "origin/main"
        }

        It "tries all candidates until one succeeds" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3)
                $null = $Command, $Args1, $Args2
                $global:LASTEXITCODE = if ($Args3 -eq 'main') { 0 } else { 1 }
                if ($global:LASTEXITCODE -eq 0) { "abc123" } else { "" }
            }
            $result = Select-DefaultBase
            $result | Should -Be "main"
        }

        It "returns null when all candidates fail" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 1; "" }
            $result = Select-DefaultBase
            $result | Should -BeNullOrEmpty
        }
    }

    Context "Get-Branch" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-Branch")
        }

        It "returns provided ref when not empty" {
            $result = Get-Branch -Ref "feature/test"
            $result | Should -Be "feature/test"
        }

        It "resolves HEAD when ref is empty" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "main" }
            $result = Get-Branch -Ref ""
            $result | Should -Be "main"
        }

        It "resolves HEAD when ref is whitespace" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "develop" }
            $result = Get-Branch -Ref "  "
            $result | Should -Be "develop"
        }
    }

    Context "Get-RemoteSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-RemoteSummary")
        }

        It "includes section header and remote output" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "origin https://github.com/test/repo (fetch)" }
            $result = Get-RemoteSummary
            $result | Should -Match "===== Repository remotes ====="
            $result | Should -Match "origin https://github.com/test/repo"
        }
    }

    Context "Get-BranchInfo" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-BranchInfo")
        }

        It "displays current branch and upstream" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3)
                $null = $Command, $Args1, $Args3
                $global:LASTEXITCODE = 0
                if ($Args2 -eq 'HEAD') { "feature/test" } else { "origin/feature/test" }
            }
            $result = Get-BranchInfo
            $result | Should -Match "feature/test"
            $result | Should -Match "origin/feature/test"
        }

        It "displays (none) when no upstream is configured" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3)
                $null = $Command, $Args1, $Args3
                if ($Args2 -eq 'HEAD') {
                    $global:LASTEXITCODE = 0
                    "main"
                } else {
                    $global:LASTEXITCODE = 1
                    ""
                }
            }
            $result = Get-BranchInfo
            $result | Should -Match "\(none\)"
        }
    }

    Context "Get-RepoStatus" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-RepoStatus")
        }

        It "includes untracked files by default" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1)
                $null = $Args1
                $global:LASTEXITCODE = 0
                if ($Command -eq 'status') { "## main" } else { "untracked.txt" }
            }
            $result = Get-RepoStatus
            $result | Should -Match "untracked.txt"
        }

        It "excludes untracked files when NoUntracked is specified" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "## main" }
            $result = Get-RepoStatus -NoUntracked
            $result | Should -Match "\(none\)"
        }

        It "displays placeholder when untracked output is empty" {
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "" }
            $result = Get-RepoStatus
            $result | Should -Match "\(none\)"
        }
    }

    Context "Get-WorkingTreeDiffSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-WorkingTreeDiffSummary")
        }

        It "includes all four diff sections" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3)
                $null = $Command, $Args3
                $global:LASTEXITCODE = 0
                if ($Args1 -eq '--cached' -and $Args2 -eq '--name-status') { "M file.txt" }
                elseif ($Args1 -eq '--cached') { "diff --git a/file.txt" }
                elseif ($Args1 -eq '--name-status') { "M other.txt" }
                else { "diff --git a/other.txt" }
            }
            $result = Get-WorkingTreeDiffSummary
            $result | Should -Match "===== Staged files"
            $result | Should -Match "===== Staged diff"
            $result | Should -Match "===== Unstaged files"
            $result | Should -Match "===== Unstaged diff"
        }
    }

    Context "Format-DiffPath" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Format-DiffPath")
        }

        It "formats brace rename paths" {
            $result = Format-DiffPath -PathText "dir/{old => new}/file.txt"
            $result | Should -Be "dir/new/file.txt"
        }

        It "handles simple rename arrow syntax" {
            $result = Format-DiffPath -PathText "old.txt => new.txt"
            $result | Should -Be "new.txt"
        }

        It "removes quotes and trims whitespace for non-whitespace input" {
            $result = Format-DiffPath -PathText '"path/to/file.txt"'
            $result | Should -Be "path/to/file.txt"
        }

        It "returns whitespace input as-is" {
            $result = Format-DiffPath -PathText "   "
            $result | Should -Be "   "
        }

        It "returns null input as-is" {
            $result = Format-DiffPath -PathText $null
            $result | Should -BeNullOrEmpty
        }

        It "handles complex brace rename with multiple segments" {
            $result = Format-DiffPath -PathText "src/{old/path => new/path}/file.cs"
            $result | Should -Be "src/new/path/file.cs"
        }
    }

    Context "ConvertFrom-Numstat" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "ConvertFrom-Numstat")
        }

        It "converts numstat text to totals and file list" {
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

        It "handles binary files with dash markers" {
            $num = @"
10	5	text.txt
-	-	binary.bin
"@
            $result = ConvertFrom-Numstat -NumstatText $num
            $result.Additions | Should -Be 10
            $result.Deletions | Should -Be 5
            $result.Files | Should -Contain "binary.bin"
        }

        It "handles empty input" {
            $result = ConvertFrom-Numstat -NumstatText ""
            $result.Additions | Should -Be 0
            $result.Deletions | Should -Be 0
            $result.Files | Should -BeNullOrEmpty
        }

        It "skips malformed lines" {
            $num = @"
5	3	good.txt
invalid line
2	1	another.txt
"@
            $result = ConvertFrom-Numstat -NumstatText $num
            $result.Additions | Should -Be 7
            $result.Deletions | Should -Be 4
            $result.Files.Count | Should -Be 2
        }
    }

    Context "Get-ExtensionSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Format-DiffPath")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ExtensionSummary")
        }

        It "summarizes extensions with counts" {
            $summary = Get-ExtensionSummary -Files @("a.ps1", "b.ps1", "c")
            $summary | Should -Match "\s+2\s+\.ps1"
            $summary | Should -Match "\s+1\s+\(noext\)"
        }

        It "handles files with no extension" {
            $summary = Get-ExtensionSummary -Files @("Makefile", "README")
            $summary | Should -Match "\s+2\s+\(noext\)"
        }

        It "handles unknown extensions gracefully" {
            $summary = Get-ExtensionSummary -Files @("test.xyz123")
            $summary | Should -Match "\.xyz123"
        }

        It "handles empty file list" {
            $summary = Get-ExtensionSummary -Files @()
            $summary | Should -Be ""
        }

        It "sorts extensions alphabetically" {
            $summary = Get-ExtensionSummary -Files @("file.z", "file.a", "file.m")
            $lines = $summary -split "`n"
            $lines[0] | Should -Match "\.a"
            $lines[2] | Should -Match "\.z"
        }
    }

    Context "Get-IssueReference" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-IssueReference")
        }

        It "collects unique issue references" {
            $refs = Get-IssueReference -Text "Fixes #12 and relates to ABC-99 plus #12 again"
            $refs | Should -Be @("#12", "ABC-99")
        }

        It "finds GitHub-style issue numbers" {
            $refs = Get-IssueReference -Text "Closes #1, #22, and #333"
            $refs.Count | Should -Be 3
            $refs | Should -Contain "#1"
            $refs | Should -Contain "#22"
            $refs | Should -Contain "#333"
        }

        It "finds JIRA-style issue keys" {
            $refs = Get-IssueReference -Text "PROJ-123 and TASK-456"
            $refs.Count | Should -Be 2
            $refs | Should -Contain "PROJ-123"
            $refs | Should -Contain "TASK-456"
        }

        It "returns empty for text with no issues" {
            $refs = Get-IssueReference -Text "No issues here"
            $refs | Should -BeNullOrEmpty
        }

        It "returns empty for null text" {
            $refs = Get-IssueReference -Text $null
            $refs | Should -BeNullOrEmpty
        }

        It "deduplicates issue references" {
            $refs = Get-IssueReference -Text "#5 #10 #5 #10"
            $refs.Count | Should -Be 2
        }

        It "does not match hash in middle of word" {
            $refs = Get-IssueReference -Text "tag#123 or word#45"
            $refs | Should -BeNullOrEmpty
        }
    }

    Context "Get-ConventionalCommitSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ConventionalCommitSummary")
        }

        It "counts conventional commit types with scope" {
            $subjects = @"
feat(scope): add feature
fix(module): bug fix
docs(readme): update readme
"@
            $result = Get-ConventionalCommitSummary -SubjectsText $subjects
            $result | Should -Match "feat\s+:\s+1"
            $result | Should -Match "fix\s+:\s+1"
            $result | Should -Match "docs\s+:\s+1"
        }

        It "counts commits with scope" {
            $subjects = "feat(api): new endpoint"
            $result = Get-ConventionalCommitSummary -SubjectsText $subjects
            $result | Should -Match "feat\s+:\s+1"
        }

        It "does not match commits without scope or exclamation before colon" {
            $subjects = @"
feat: breaking change
fix: simple fix
"@
            $result = Get-ConventionalCommitSummary -SubjectsText $subjects
            $result | Should -Match "other\s+:\s+2"
        }

        It "categorizes non-conventional commits as other" {
            $subjects = @"
Update file
Random commit message
"@
            $result = Get-ConventionalCommitSummary -SubjectsText $subjects
            $result | Should -Match "other\s+:\s+2"
        }

        It "returns placeholder for empty input" {
            $result = Get-ConventionalCommitSummary -SubjectsText ""
            $result | Should -Match "no recognizable"
        }

        It "supports all conventional commit types with scopes" {
            $subjects = @"
feat(x): feature
fix(x): fix
refactor(x): refactor
perf(x): performance
docs(x): documentation
test(x): test
chore(x): chore
build(x): build
ci(x): ci
style(x): style
"@
            $result = Get-ConventionalCommitSummary -SubjectsText $subjects
            $result | Should -Match "feat"
            $result | Should -Match "fix"
            $result | Should -Match "refactor"
            $result | Should -Match "perf"
            $result | Should -Match "docs"
            $result | Should -Match "test"
            $result | Should -Match "chore"
            $result | Should -Match "build"
            $result | Should -Match "ci"
            $result | Should -Match "style"
        }
    }

    Context "Get-PRContext" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "ConvertFrom-Numstat")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Format-DiffPath")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ExtensionSummary")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-IssueReference")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ConventionalCommitSummary")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ItemCount")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-PRContext")
        }

        It "generates complete PR context with valid base and head" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3, $Args4, $Args5)
                $null = $Args3, $Args4, $Args5
                $global:LASTEXITCODE = 0
                if ($Command -eq 'rev-parse') { "abc123" }
                elseif ($Command -eq 'merge-base') { "def456" }
                elseif ($Command -eq 'log' -and $Args2 -like '*%h*') { "abc123 2025-01-01 Author Test commit" }
                elseif ($Command -eq 'log' -and $Args2 -like '*%s') { "feat: test" }
                elseif ($Command -eq 'log' -and $Args2 -like '*%an*') { "Author <author@example.com>" }
                elseif ($Command -eq 'diff' -and $Args1 -eq '--name-status') { "M file.txt" }
                elseif ($Command -eq 'diff' -and $Args1 -eq '--numstat') { "1`t0`tfile.txt" }
                elseif ($Command -eq 'diff' -and $Args1 -eq '--shortstat') { "1 file changed, 1 insertion(+)" }
                elseif ($Command -eq 'diff' -and $Args1 -eq '--stat') { "file.txt | 1 +`n 1 file changed, 1 insertion(+)" }
                else { "" }
            }
            $result = Get-PRContext -BaseRef "main" -HeadRef "feature"
            $result | Should -Match "===== PR Comparison ====="
            $result | Should -Match "Base: main"
            $result | Should -Match "Head: feature"
            $result | Should -Match "===== Commits in range ====="
            $result | Should -Match "===== Authors ====="
            $result | Should -Match "===== Changed files"
        }

        It "includes issue references when found in commits" {
            Mock -CommandName git -MockWith {
                param($Command, $Args1, $Args2, $Args3, $Args4, $Args5)
                $null = $Args3, $Args4, $Args5
                $global:LASTEXITCODE = 0
                if ($Command -eq 'rev-parse') { "abc123" }
                elseif ($Command -eq 'merge-base') { "def456" }
                elseif ($Command -eq 'log' -and $Args2 -like '*%s') { "fix: #42 resolved" }
                elseif ($Command -eq 'log') { "abc123 2025-01-01 Author Fix #42" }
                elseif ($Command -eq 'diff' -and $Args1 -eq '--numstat') { "1`t0`tfile.txt" }
                else { "" }
            }
            $result = Get-PRContext -BaseRef "main" -HeadRef "feature"
            $result | Should -Match "#42"
        }

        It "displays placeholders for empty results" {
            Mock -CommandName git -MockWith {
                param($Command)
                $global:LASTEXITCODE = 0
                if ($Command -eq 'rev-parse') { "abc123" }
                elseif ($Command -eq 'merge-base') { "def456" }
                else { "" }
            }
            $result = Get-PRContext -BaseRef "main" -HeadRef "feature"
            $result | Should -Match "\(none\)"
        }
    }

    Context "Resolve-Repo" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Resolve-Repo")
        }

        It "stays in directory when .git exists" {
            Mock -CommandName Test-Path -MockWith { $true }
            Mock -CommandName Push-Location -MockWith { }
            { Resolve-Repo -Root "." } | Should -Not -Throw
        }

        It "navigates to git toplevel when .git is missing" {
            Mock -CommandName Test-Path -MockWith { $false }
            Mock -CommandName Push-Location -MockWith { }
            Mock -CommandName Pop-Location -MockWith { }
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 0; "/repo/root" }
            { Resolve-Repo -Root "." } | Should -Not -Throw
        }

        It "throws error when not in a git repo" {
            Mock -CommandName Test-Path -MockWith { $false }
            Mock -CommandName Push-Location -MockWith { }
            Mock -CommandName Pop-Location -MockWith { }
            Mock -CommandName git -MockWith { $global:LASTEXITCODE = 1; "" }
            { Resolve-Repo -Root "." } | Should -Throw "*git*failed*"
        }
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

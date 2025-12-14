Set-StrictMode -Version Latest

Describe "collect-pull-request-context.ps1" {
    BeforeAll {
        $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot} else { Split-Path -Parent $PSCommandPath }
        . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../../powershell/Support/TestHelpers.ps1"))
        
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\collect-pull-request-context.ps1"
    }

    Context "Invoke-Git" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
        }

        It "executes git command successfully with string output" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; "output" }
            $result = Invoke-Git -GitArgs @('status')
            $result.Out | Should -Be "output"
            $result.Code | Should -Be 0
        }

        It "executes git command successfully with array output" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; @("line1", "line2") }
            $result = Invoke-Git -GitArgs @('log')
            $result.Out | Should -Be "line1`nline2"
            $result.Code | Should -Be 0
        }

        It "handles null output" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; $null }
            $result = Invoke-Git -GitArgs @('status')
            $result.Out | Should -Be ""
            $result.Code | Should -Be 0
        }

        It "throws on non-zero exit when AllowNonZeroExit is false" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 1; "error output" }
            { Invoke-Git -GitArgs @('invalid') } | Should -Throw "git invalid failed*"
        }

        It "does not throw on non-zero exit when AllowNonZeroExit is true" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 1; "error output" }
            $result = Invoke-Git -GitArgs @('invalid') -AllowNonZeroExit
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
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Select-DefaultBase")
        }

        It "returns first valid ref from candidate list" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                # Expected call shape: rev-parse --verify --quiet <ref>
                $ref = $GitArgs[-1]
                $global:LASTEXITCODE = if ($ref -eq 'origin/main') { 0 } else { 1 }
                if ($global:LASTEXITCODE -eq 0) { "abc123" } else { "" }
            }

            $result = Select-DefaultBase
            $result | Should -Be "origin/main"
        }

        It "tries all candidates until one succeeds" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                $ref = $GitArgs[-1]
                $global:LASTEXITCODE = if ($ref -eq 'main') { 0 } else { 1 }
                if ($global:LASTEXITCODE -eq 0) { "abc123" } else { "" }
            }

            $result = Select-DefaultBase
            $result | Should -Be "main"
        }

        It "returns null when all candidates fail" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)
                $null = $GitArgs
                $global:LASTEXITCODE = 1
                ""
            }

            $result = Select-DefaultBase
            $result | Should -BeNullOrEmpty
        }
    }

    Context "Get-Branch" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-Branch")
        }

        It "returns provided ref when not empty" {
            $result = Get-Branch -Ref "feature/test"
            $result | Should -Be "feature/test"
        }

        It "resolves HEAD when ref is empty" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; "main" }
            $result = Get-Branch -Ref ""
            $result | Should -Be "main"
        }

        It "resolves HEAD when ref is whitespace" {
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; "develop" }
            $result = Get-Branch -Ref "  "
            $result | Should -Be "develop"
        }
    }

    Context "Get-RemoteSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-RemoteSummary")
        }

        It "includes section header and remote output" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)
                $global:LASTEXITCODE = 0
                if ($GitArgs[0] -eq 'remote') { "origin https://github.com/test/repo (fetch)" } else { "" }
            }

            $result = Get-RemoteSummary
            $result | Should -Match "===== Repository remotes ====="
            $result | Should -Match "origin https://github.com/test/repo"
        }
    }

    Context "Get-BranchInfo" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-BranchInfo")
        }

        It "displays current branch and upstream" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                # rev-parse --abbrev-ref HEAD
                # rev-parse --abbrev-ref --symbolic-full-name @{u}
                $global:LASTEXITCODE = 0

                if ($GitArgs[-1] -eq 'HEAD') { "feature/test" }
                else { "origin/feature/test" }
            }

            $result = Get-BranchInfo
            $result | Should -Match "feature/test"
            $result | Should -Match "origin/feature/test"
        }

        It "displays (none) when no upstream is configured" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                if ($GitArgs[-1] -eq 'HEAD') {
                    $global:LASTEXITCODE = 0
                    "main"
                }
                else {
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
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-RepoStatus")
        }

        It "includes untracked files by default" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)
                $global:LASTEXITCODE = 0

                if ($GitArgs[0] -eq 'status') { "## main" }
                elseif ($GitArgs[0] -eq 'ls-files') { "untracked.txt" }
                else { "" }
            }

            $result = Get-RepoStatus
            $result | Should -Match "untracked.txt"
        }

        It "excludes untracked files when NoUntracked is specified" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)
                $null = $GitArgs
                $global:LASTEXITCODE = 0
                "## main"
            }

            $result = Get-RepoStatus -NoUntracked
            $result | Should -Match "\(none\)"
        }

        It "displays placeholder when untracked output is empty" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)
                $global:LASTEXITCODE = 0

                if ($GitArgs[0] -eq 'status') { "## main" }
                elseif ($GitArgs[0] -eq 'ls-files') { "" }
                else { "" }
            }

            $result = Get-RepoStatus
            $result | Should -Match "\(none\)"
        }
    }

    Context "Get-WorkingTreeDiffSummary" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Write-Section")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-WorkingTreeDiffSummary")
        }

        It "includes all four diff sections" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                # diff --cached --name-status
                # diff --cached
                # diff --name-status
                # diff
                $global:LASTEXITCODE = 0

                if ($GitArgs[0] -ne 'diff') { return "" }

                $hasCached = ($GitArgs | Where-Object { $_ -eq '--cached' }).Count -gt 0
                $hasNameStatus = ($GitArgs | Where-Object { $_ -eq '--name-status' }).Count -gt 0

                if ($hasCached -and $hasNameStatus) { "M file.txt" }
                elseif ($hasCached) { "diff --git a/file.txt" }
                elseif ($hasNameStatus) { "M other.txt" }
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
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
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
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                $cmd = if ($GitArgs.Count -gt 0) { $GitArgs[0] } else { "" }
                $global:LASTEXITCODE = 0

                if ($cmd -eq 'rev-parse') { "abc123" }
                elseif ($cmd -eq 'merge-base') { "def456" }
                elseif ($cmd -eq 'log' -and ($GitArgs | Where-Object { $_ -like '*%h*' }).Count -gt 0) { "abc123 2025-01-01 Author Test commit" }
                elseif ($cmd -eq 'log' -and ($GitArgs | Where-Object { $_ -like '*%s' }).Count -gt 0) { "feat: test" }
                elseif ($cmd -eq 'log' -and ($GitArgs | Where-Object { $_ -like '*%an*' }).Count -gt 0) { "Author <author@example.com>" }
                elseif ($cmd -eq 'diff' -and ($GitArgs | Where-Object { $_ -eq '--name-status' }).Count -gt 0) { "M file.txt" }
                elseif ($cmd -eq 'diff' -and ($GitArgs | Where-Object { $_ -eq '--numstat' }).Count -gt 0) { "1`t0`tfile.txt" }
                elseif ($cmd -eq 'diff' -and ($GitArgs | Where-Object { $_ -eq '--shortstat' }).Count -gt 0) { "1 file changed, 1 insertion(+)" }
                elseif ($cmd -eq 'diff' -and ($GitArgs | Where-Object { $_ -eq '--stat' }).Count -gt 0) { "file.txt | 1 +`n 1 file changed, 1 insertion(+)" }
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
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                $cmd = if ($GitArgs.Count -gt 0) { $GitArgs[0] } else { "" }
                $global:LASTEXITCODE = 0

                if ($cmd -eq 'rev-parse') { "abc123" }
                elseif ($cmd -eq 'merge-base') { "def456" }
                elseif ($cmd -eq 'log' -and ($GitArgs | Where-Object { $_ -like '*%s' }).Count -gt 0) { "fix: #42 resolved" }
                elseif ($cmd -eq 'log') { "abc123 2025-01-01 Author Fix #42" }
                elseif ($cmd -eq 'diff' -and ($GitArgs | Where-Object { $_ -eq '--numstat' }).Count -gt 0) { "1`t0`tfile.txt" }
                else { "" }
            }

            $result = Get-PRContext -BaseRef "main" -HeadRef "feature"
            $result | Should -Match "#42"
        }

        It "displays placeholders for empty results" {
            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                $cmd = if ($GitArgs.Count -gt 0) { $GitArgs[0] } else { "" }
                $global:LASTEXITCODE = 0

                if ($cmd -eq 'rev-parse') { "abc123" }
                elseif ($cmd -eq 'merge-base') { "def456" }
                else { "" }
            }

            $result = Get-PRContext -BaseRef "main" -HeadRef "feature"
            $result | Should -Match "\(none\)"
        }
    }

    Context "Resolve-Repo" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
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
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 0; "/repo/root" }

            { Resolve-Repo -Root "." } | Should -Not -Throw
        }

        It "throws error when not in a git repo" {
            Mock -CommandName Test-Path -MockWith { $false }
            Mock -CommandName Push-Location -MockWith { }
            Mock -CommandName Pop-Location -MockWith { }
            Mock -CommandName Invoke-GitExe -MockWith { param([string[]]$GitArgs) $null = $GitArgs; $global:LASTEXITCODE = 1; "" }

            { Resolve-Repo -Root "." } | Should -Throw "*git*failed*"
        }
    }
}

Set-StrictMode -Version Latest

Describe "collect-commit-context.ps1" {
    BeforeAll {
        $script:scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
        $script:helperPath = Join-Path -Path $script:scriptRoot -ChildPath "..\Support\TestHelpers.ps1"
        . (Resolve-Path -Path $script:helperPath)
        $script:collectCommitScript = Join-Path -Path $script:scriptRoot -ChildPath "..\..\..\scripts\dev-tools\collect-commit-context.ps1"
    }

    Context "Add-ReportSection function" {
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
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "Test Section" -Cmd { "line1`nline2" }

            $script:captured.Count | Should -Be 2
            $script:captured[0] | Should -Match "===== Test Section ====="
            $script:captured[1] | Should -Match "line1`nline2"
        }

        It "writes placeholder when allowed to fail" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "MayFail" -Cmd { throw "boom" } -AllowFail
            $script:captured[1] | Should -Match "\[n/a\]"
        }

        It "throws error when not allowed to fail" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            { Add-ReportSection -Title "MustSucceed" -Cmd { throw "error" } } | Should -Throw
        }

        It "trims trailing whitespace from command output" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "Trimmed" -Cmd { "content   `n  " }

            $script:captured[1] | Should -Be "content"
        }

        It "handles empty command output" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "Empty" -Cmd { "" }

            $script:captured.Count | Should -Be 2
            $script:captured[0] | Should -Match "===== Empty ====="
        }

        It "handles null command parameter" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "NoCmd" -Cmd $null

            $script:captured.Count | Should -Be 1
            $script:captured[0] | Should -Match "===== NoCmd ====="
        }

        It "writes multiple sections to same output" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "First" -Cmd { "content1" }
            Add-ReportSection -Title "Second" -Cmd { "content2" }

            $script:captured.Count | Should -Be 4
            $script:captured[0] | Should -Match "===== First ====="
            $script:captured[2] | Should -Match "===== Second ====="
        }

        It "handles multiline output correctly" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            $multiline = "line1`nline2`nline3"
            Add-ReportSection -Title "MultiLine" -Cmd { $multiline }

            $script:captured[1] | Should -Be $multiline
        }

        It "handles scriptblock that returns objects" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "Objects" -Cmd { [PSCustomObject]@{ Name = "test"; Value = 42 } }

            $script:captured.Count | Should -Be 2
            $script:captured[1] | Should -Match "test"
        }

        It "handles command with no output" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            Add-ReportSection -Title "NoOutput" -Cmd { $null }

            $script:captured.Count | Should -Be 2
            $script:captured[1] | Should -Be ""
        }

        It "preserves line breaks in multiline content" {
            . (Import-ScriptFunction -Path $script:collectCommitScript -Name "Add-ReportSection")
            $content = @"
line1
line2
line3
"@
            Add-ReportSection -Title "Lines" -Cmd { $content }

            $script:captured[1] | Should -Match "line1"
            $script:captured[1] | Should -Match "line2"
            $script:captured[1] | Should -Match "line3"
        }
    }

    Context "Script execution" {
        # Use mocks to keep the test isolated from Git and the filesystem per unit-test policy.
        BeforeEach {
            $gitCallLog = New-Object System.Collections.Generic.List[string]
            $addContentCalls = New-Object System.Collections.Generic.List[hashtable]
            $createdDirs = New-Object System.Collections.Generic.List[string]
            $removedPaths = New-Object System.Collections.Generic.List[string]
            $setLocations = New-Object System.Collections.Generic.List[string]

            $dirExists = $true
            $fileExists = $false

            $expectedRoot = "C:\repo-root"
            $outputPath = Join-Path -Path $expectedRoot -ChildPath "artifacts/commit_context.txt"
            $outputDir = Split-Path -Parent $outputPath

            $gitOutputs = New-Object 'System.Collections.Generic.Dictionary[string,string]'
            $gitOutputs.Add('rev-parse --is-inside-work-tree', 'true')
            $gitOutputs.Add('rev-parse --show-toplevel', $expectedRoot)
            $gitOutputs.Add('remote -v', "origin https://example/repo (fetch)`norigin https://example/repo (push)")
            $gitOutputs.Add('branch --show-current', 'feature/branch')
            $gitOutputs.Add('rev-parse --abbrev-ref --symbolic-full-name @{u}', 'origin/feature/branch')
            $gitOutputs.Add('status -sb', ' M file.ps1')
            $gitOutputs.Add('diff --staged --name-status', "M`tfile.ps1")
            $gitOutputs.Add('diff --staged', 'staged diff')
            $gitOutputs.Add('diff --name-status', "M`tfile.ps1")
            $gitOutputs.Add('diff', 'unstaged diff')
            $gitOutputs.Add('ls-files --others --exclude-standard', 'untracked.txt')
            $gitOutputs.Add('diff --numstat', "1`t2`tfile1.ps1")
            $gitOutputs.Add('diff --staged --numstat', "3`t4`tfile2.ps1")
            $gitOutputs.Add('diff --name-only HEAD -- *.py', "a.py`nb.py")
            $gitOutputs.Add('diff --name-only HEAD *.py', "a.py`nb.py")
            $gitOutputs.Add('show -s --pretty=fuller -1', 'commit header')

            Mock -CommandName git -MockWith {
                $call = ($args -join ' ')
                $gitCallLog.Add($call)
                if (-not $gitOutputs.ContainsKey($call)) {
                    throw "Unexpected git call: $call"
                }

                return $gitOutputs[$call]
            }

            Mock -CommandName Test-Path -MockWith {
                param($Path)
                if ($Path -eq $outputDir) { return $dirExists }
                if ($Path -eq $outputPath) { return $fileExists }
                return $true
            }

            Mock -CommandName New-Item -MockWith {
                param($ItemType, $Path, $Force)
                $null = $ItemType
                $null = $Force
                $createdDirs.Add($Path)
            }

            Mock -CommandName Remove-Item -MockWith {
                param($Path, $Force)
                $null = $Force
                $removedPaths.Add($Path)
            }

            Mock -CommandName Add-Content -MockWith {
                param($Path, $Value)
                $addContentCalls.Add(@{ Path = $Path; Value = $Value })
            }

            Mock -CommandName Set-Location -MockWith {
                param($Path)
                $setLocations.Add($Path)
            }
        }

        It "writes report sections using git output" {
            $outputText = & $script:collectCommitScript 2>$null | Out-String

            $setLocations | Should -Contain $expectedRoot
            ($addContentCalls | ForEach-Object { $_.Path } | Select-Object -Unique) | Should -Be @($outputPath)

            $addContentCalls[0].Value | Should -Match "^Please generate a commit message"
            ($addContentCalls | Where-Object { $_.Value -like "*===== Repository remotes*" }).Count | Should -BeGreaterThan 0
            ($addContentCalls | Where-Object { $_.Value -like "*origin https://example/repo*" }).Count | Should -BeGreaterThan 0
            ($addContentCalls | Where-Object { $_.Value -like "*feature/branch*" }).Count | Should -BeGreaterThan 0

            $gitCallLog | Should -Contain 'remote -v'
            $gitCallLog | Should -Contain 'branch --show-current'
            $outputText | Should -Match ([regex]::Escape($outputPath))
        }

        It "creates output directory when missing" {
            $dirExists = $false
            $dirExists | Should -BeFalse

            & $script:collectCommitScript | Out-Null

            $createdDirs | Should -Contain $outputDir
        }

        It "removes existing output file before writing" {
            $dirExists = $true
            $fileExists = $true
            $dirExists | Should -BeTrue
            $fileExists | Should -BeTrue

            & $script:collectCommitScript | Out-Null

            $removedPaths | Should -Contain $outputPath
            ($addContentCalls | ForEach-Object { $_.Path } | Select-Object -Unique) | Should -Be @($outputPath)
        }
    }
}


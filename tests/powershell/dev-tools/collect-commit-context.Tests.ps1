Set-StrictMode -Version Latest

Describe "collect-commit-context.ps1" {
    BeforeAll {
        $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
        . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../Support/TestHelpers.ps1"))

        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\collect-commit-context.ps1"
    }

    Context "Add-ReportSection function" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Add-ReportSection")
            $script:ReportOutput = "mock-output.txt"
            $script:captured = New-Object System.Collections.Generic.List[string]
        }

        It "writes section headers and content" {
            $mockAddContent = {
                param($Path, $Value)
                $null = $Path
                $script:captured.Add($Value)
            }

            Add-ReportSection -Title "Test Section" -Cmd { "line1`nline2" } -AddContentFunc $mockAddContent

            $script:captured.Count | Should -Be 2
            $script:captured[0] | Should -Match "===== Test Section ====="
            $script:captured[1] | Should -Match "line1`nline2"
        }

        It "writes placeholder when allowed to fail" {
            $mockAddContent = {
                param($Path, $Value)
                $null = $Path
                $script:captured.Add($Value)
            }

            Add-ReportSection -Title "MayFail" -Cmd { throw "boom" } -AllowFail -AddContentFunc $mockAddContent
            $script:captured[1] | Should -Match "\[n/a\]"
        }

        It "throws error when not allowed to fail" {
            $mockAddContent = {
                param($Path, $Value)
                $null = $Path
                $script:captured.Add($Value)
            }

            { Add-ReportSection -Title "MustSucceed" -Cmd { throw "error" } -AddContentFunc $mockAddContent } | Should -Throw
        }
    }

    Context "Script execution" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-GitExe")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-Git")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Add-ReportSection")
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-CollectCommitContext")

            $gitCallLog = New-Object System.Collections.Generic.List[string]
            $addContentCalls = New-Object System.Collections.Generic.List[hashtable]
            $createdDirs = New-Object System.Collections.Generic.List[string]
            $removedPaths = New-Object System.Collections.Generic.List[string]
            $setLocations = New-Object System.Collections.Generic.List[string]

            $dirExists = $true
            $fileExists = $false

            $expectedRoot = "C:\repo-root"
            $outputPath = "C:\repo-root\artifacts\commit_context.txt"
            $outputDir = "C:\repo-root\artifacts"

            Mock -CommandName Invoke-GitExe -MockWith {
                param([string[]]$GitArgs)

                $call = ($GitArgs -join ' ')
                $gitCallLog.Add($call)

                $global:LASTEXITCODE = 0

                switch ($call) {
                    'rev-parse --is-inside-work-tree' { return 'true' }
                    'rev-parse --show-toplevel' { return $expectedRoot }
                    'remote -v' { return "origin https://example/repo (fetch)`norigin https://example/repo (push)" }
                    'branch --show-current' { return 'feature/branch' }
                    'rev-parse --abbrev-ref --symbolic-full-name @{u}' { return 'origin/feature/branch' }
                    'status -sb' { return ' M file.ps1' }
                    'diff --staged --name-status' { return "M`tfile.ps1" }
                    'diff --staged' { return 'staged diff' }
                    'diff --name-status' { return "M`tfile.ps1" }
                    'diff' { return 'unstaged diff' }
                    'ls-files --others --exclude-standard' { return 'untracked.txt' }
                    'diff --numstat' { return "1`t2`tfile1.ps1" }
                    'diff --staged --numstat' { return "3`t4`tfile2.ps1" }
                    'diff --name-only HEAD -- *.py' { return "a.py`nb.py" }
                    'show -s --pretty=fuller -1' { return 'commit header' }
                    default {
                        throw "Unexpected git call: $call"
                    }
                }
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

        AfterEach {
            $gitCallLog.Clear()
            $addContentCalls.Clear()
            $createdDirs.Clear()
            $removedPaths.Clear()
            $setLocations.Clear()
        }

        It "writes report sections using git output" {
            $outputText = Invoke-CollectCommitContext -Output "artifacts/commit_context.txt" 2>$null | Out-String

            $setLocations | Should -Contain $expectedRoot
            ($addContentCalls | ForEach-Object { $_.Path } | Select-Object -Unique) | Should -Be @($outputPath)

            $allValues = $addContentCalls | ForEach-Object { $_.Value }
            $allValues | Where-Object { $_ -match "^Please generate a commit message" } | Should -Not -BeNullOrEmpty
            ($addContentCalls | Where-Object { $_.Value -like "*===== Repository remotes*" }).Count | Should -BeGreaterThan 0
            ($addContentCalls | Where-Object { $_.Value -like "*origin https://example/repo*" }).Count | Should -BeGreaterThan 0
            ($addContentCalls | Where-Object { $_.Value -like "*feature/branch*" }).Count | Should -BeGreaterThan 0

            $gitCallLog | Should -Contain 'remote -v'
            $gitCallLog | Should -Contain 'branch --show-current'
            $gitCallLog | Should -Contain 'rev-parse --is-inside-work-tree'
            $outputText | Should -Match ([regex]::Escape($outputPath))
        }

        It "creates output directory when missing" {
            $dirExists = $false
            $dirExists | Should -BeFalse

            Invoke-CollectCommitContext -Output "artifacts/commit_context.txt" | Out-Null

            $createdDirs | Should -Contain $outputDir
        }

        It "removes existing output file before writing" {
            $dirExists = $true
            $fileExists = $true
            $dirExists | Should -BeTrue
            $fileExists | Should -BeTrue

            Invoke-CollectCommitContext -Output "artifacts/commit_context.txt" | Out-Null

            $removedPaths | Should -Contain $outputPath
            ($addContentCalls | ForEach-Object { $_.Path } | Select-Object -Unique) | Should -Be @($outputPath)
        }
    }
}

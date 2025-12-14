Set-StrictMode -Version Latest

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
. (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "Support/TestHelpers.ps1"))

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
        $num = "4`t2`tfile1.ps1`n1`t0`tdir/file2.psm1"
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

Describe "new-potential-entry.ps1 validation" {
    It "contains short name validation pattern" {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\new-potential-entry.ps1"
        (Get-Content -Path $scriptPath -Raw) | Should -Match "\^\[a-z0-9\]\+\(-\[a-z0-9\]\+\)\*\$"
    }
}

Describe "potential-to-issue.ps1 helpers" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
    }

    Context "Get-FeatureName function" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-FeatureName")
        }

        It "extracts feature name from markdown heading" {
            $content = "# My Feature Name`n## Section"
            $result = Get-FeatureName -Content $content -FilePath "test.md"
            $result | Should -Be "My Feature Name"
        }

        It "removes (Potential) suffix from heading" {
            $content = "# Feature (Potential)`n## Section"
            $result = Get-FeatureName -Content $content -FilePath "test.md"
            $result | Should -Be "Feature"
        }

        It "trims whitespace after removing (Potential)" {
            $content = "#   Feature Name (Potential)  `n## Section"
            $result = Get-FeatureName -Content $content -FilePath "test.md"
            $result | Should -Be "Feature Name"
        }

        It "falls back to filename when no heading found" {
            $content = "No heading here"
            $result = Get-FeatureName -Content $content -FilePath "C:\path\to\my-feature.md"
            $result | Should -Be "my-feature"
        }

        It "removes .md extension from filename fallback" {
            $content = "No heading"
            $result = Get-FeatureName -Content $content -FilePath "feature-name.md"
            $result | Should -Be "feature-name"
        }

        It "handles heading with special characters" {
            $content = "# Feature: Advanced (v2.0)`n## Section"
            $result = Get-FeatureName -Content $content -FilePath "test.md"
            $result | Should -Be "Feature: Advanced (v2.0)"
        }

        It "uses first heading when multiple exist" {
            $content = "# First Feature`n## Second`n# Third"
            $result = Get-FeatureName -Content $content -FilePath "test.md"
            $result | Should -Be "First Feature"
        }
    }

    Context "Get-FeaturePath function" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-FeaturePath")
        }

        It "replaces spaces with underscores" {
            $result = Get-FeaturePath -FeatureName "My Feature Name"
            $result | Should -Be "My_Feature_Name"
        }

        It "removes special characters except alphanumeric, underscore, and hyphen" {
            $result = Get-FeaturePath -FeatureName "Feature: (v2.0) @ Test!"
            $result | Should -Be "Feature_v20__Test"
        }

        It "handles multiple consecutive spaces" {
            $result = Get-FeaturePath -FeatureName "Feature   Name"
            $result | Should -Be "Feature_Name"
        }

        It "preserves hyphens in feature name" {
            $result = Get-FeaturePath -FeatureName "my-feature-name"
            $result | Should -Be "my-feature-name"
        }

        It "handles feature name with numbers" {
            $result = Get-FeaturePath -FeatureName "Feature v2 Update"
            $result | Should -Be "Feature_v2_Update"
        }

        It "handles single character feature name" {
            $result = Get-FeaturePath -FeatureName "A"
            $result | Should -Be "A"
        }
    }

    Context "Get-Section function" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-Section")
        }

        It "extracts a section by heading" {
            $script:content = "## Problem / Why`nabc`n## Proposed Behavior`ndef"
            $result = Get-Section -Name "Problem / Why"
            $result | Should -Be "abc"
        }

        It "extracts section with multiple lines" {
            $script:content = "## Problem / Why`nline1`nline2`nline3`n## Next Section`nother"
            $result = Get-Section -Name "Problem / Why"
            $result | Should -Be "line1`nline2`nline3"
        }

        It "returns empty string when section not found" {
            $script:content = "## Problem / Why`nabc`n## Proposed Behavior`ndef"
            $result = Get-Section -Name "NonExistent"
            $result | Should -Be ""
        }

        It "handles section at end of document" {
            $script:content = "## Problem / Why`nabc`n## Last Section`nfinal content"
            $result = Get-Section -Name "Last Section"
            $result | Should -Be "final content"
        }

        It "trims whitespace from section content" {
            $script:content = "## Problem / Why`n  abc  `n  def  `n## Next"
            $result = Get-Section -Name "Problem / Why"
            $result | Should -Be "abc  `n  def"
        }

        It "handles sections with special characters in heading" {
            $script:content = "## Acceptance Criteria (early draft)`ncontent here`n## Next"
            $result = Get-Section -Name "Acceptance Criteria (early draft)"
            $result | Should -Be "content here"
        }

        It "handles empty section" {
            $script:content = "## Problem / Why`n`n## Proposed Behavior`ndef"
            $result = Get-Section -Name "Problem / Why"
            $result | Should -Be ""
        }

        It "handles section with windows line endings" {
            $script:content = "## Problem / Why`r`nabc`r`n## Proposed Behavior`r`ndef"
            $result = Get-Section -Name "Problem / Why"
            $result | Should -Be "abc"
        }
    }

    Context "Set-LineValue function" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Set-LineValue")
        }

        It "inserts new metadata line when label not found" {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]]@("# Title", "- Issue: #1"))
            $metaEnd = 2
            $metaRef = [ref]$metaEnd

            Set-LineValue -arr $lines -label "Issue URL" -value "https://example" -metaEndRef $metaRef
            $lines | Should -Contain "- Issue URL: https://example"
            $metaRef.Value | Should -Be 3
        }

        It "updates existing metadata line when label found" {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]]@("# Title", "- Issue: #1"))
            $metaEnd = 2
            $metaRef = [ref]$metaEnd

            Set-LineValue -arr $lines -label "Issue" -value "#2" -metaEndRef $metaRef
            ($lines | Where-Object { $_ -like "- Issue:*" }) | Should -Contain "- Issue: #2"
            $metaRef.Value | Should -Be 2
        }

        It "inserts at correct position using metaEnd" {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]]@("# Title", "## First Section", "content"))
            $metaEnd = 1
            $metaRef = [ref]$metaEnd

            Set-LineValue -arr $lines -label "Status" -value "Active" -metaEndRef $metaRef
            $lines[1] | Should -Be "- Status: Active"
            $lines[2] | Should -Be "## First Section"
        }

        It "handles multiple insertions incrementing metaEnd" {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]]@("# Title"))
            $metaEnd = 1
            $metaRef = [ref]$metaEnd

            Set-LineValue -arr $lines -label "Issue" -value "#1" -metaEndRef $metaRef
            Set-LineValue -arr $lines -label "URL" -value "http://test" -metaEndRef $metaRef
            Set-LineValue -arr $lines -label "Status" -value "Active" -metaEndRef $metaRef

            $lines | Should -Contain "- Issue: #1"
            $lines | Should -Contain "- URL: http://test"
            $lines | Should -Contain "- Status: Active"
            $metaRef.Value | Should -Be 4
        }

        It "respects WhatIf when using ShouldProcess" {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]]@("# Title"))
            $metaEnd = 1
            $metaRef = [ref]$metaEnd

            Set-LineValue -arr $lines -label "Test" -value "Value" -metaEndRef $metaRef -WhatIf
            $lines | Should -Not -Contain "- Test: Value"
        }
    }

    Context "Write-ScriptError function" {
        It "is a CmdletBinding function that accepts a Message parameter" {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
            $scriptContent = Get-Content -Path $scriptPath -Raw
            $scriptContent | Should -Match "function Write-ScriptError"
            $scriptContent | Should -Match "\[CmdletBinding\(\)\]"
            $scriptContent | Should -Match "param\(\s+\[Parameter\(Mandatory = \`$true\)\]\s+\[string\] \`$Message"
        }

        It "calls Write-Error in the function body" {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
            $scriptContent = Get-Content -Path $scriptPath -Raw
            # Verify the function contains Write-Error command
            $scriptContent | Should -Match "function Write-ScriptError[\s\S]+?Write-Error.*Message"
        }

        It "calls exit 1 in the function body" {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\potential-to-issue.ps1"
            $scriptContent = Get-Content -Path $scriptPath -Raw
            # Verify the function contains exit 1 command
            $scriptContent | Should -Match "function Write-ScriptError[\s\S]+?exit 1"
        }
    }
}

Describe "run-cloc.ps1" {
    BeforeAll {
        [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseDeclaredVarsMoreThanAssignments', 'scriptPath')]
        param()
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\scripts\dev-tools\run-cloc.ps1"  # type: ignore
    }

    Context "Initialize-OutputRendering" {
        It "sets PSStyle.OutputRendering to PlainText on PowerShell 7+" -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
            . (Import-ScriptFunction -Path $scriptPath -Name "Initialize-OutputRendering")
            $originalValue = $PSStyle.OutputRendering
            try {
                $PSStyle.OutputRendering = 'Ansi'
                Initialize-OutputRendering
                $PSStyle.OutputRendering | Should -Be 'PlainText'
            }
            finally {
                $PSStyle.OutputRendering = $originalValue
            }
        }

        It "does not error on Windows PowerShell 5.1" {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Initialize-OutputRendering")
            { Initialize-OutputRendering } | Should -Not -Throw
        }
    }

    Context "Test-IsWindows" {
        It "returns true on Windows PowerShell 5.1 when OS is Windows_NT" -Skip:($PSVersionTable.PSVersion.Major -ge 6) {
            . (Import-ScriptFunction -Path $scriptPath -Name "Test-IsWindows")
            $result = Test-IsWindows
            $result | Should -BeOfType [bool]
        }

        It "returns platform detection on PowerShell 6+" -Skip:($PSVersionTable.PSVersion.Major -lt 6) {
            . (Import-ScriptFunction -Path $scriptPath -Name "Test-IsWindows")
            $result = Test-IsWindows
            $result | Should -Be $IsWindows
        }
    }

    Context "Get-ClocPath" {
        It "constructs correct paths from script root and target path" {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ClocPath")
            $resolver = { param($InputPath) [pscustomobject]@{ Path = "C\resolved\$InputPath" } }

            $result = Get-ClocPath -ScriptRoot "C\script" -TargetPath "target" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\resolved\target"
            $result.ClocExe | Should -Match "tools\\cloc\.exe$"
            $result.ClocScript | Should -Match "tools\\cloc$"
        }
        It "resolves relative target paths" {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ClocPath")
            $resolver = {
                [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', '')]
                param($InputPath)
                $null = $InputPath
                [pscustomobject]@{ Path = "C\absolute\path" }
            }

            $result = Get-ClocPath -ScriptRoot "C\base" -TargetPath "../relative" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\absolute\path"
        }
        Context "Invoke-ClocCount" {
            BeforeEach {
                . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-ClocCount")
                $script:executedCommand = $null
                $script:executedArgs = $null
            }

            It "runs cloc.exe on Windows when it exists" {
                $paths = @{}
                $paths.Root = "C\\repo"
                $paths.ClocExe = "C\\tools\cloc.exe"
                $paths.ClocScript = "C\\tools\cloc"

                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $true }
                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $false }

                $global:LASTEXITCODE = 0
                $runner = {
                    param($Command, $Arguments)
                    $script:executedCommand = $Command
                    $script:executedArgs = $Arguments
                }

                { Invoke-ClocCount -Paths $paths -IsWindows $true -InvokeProcess $runner } | Should -Not -Throw
                $script:executedCommand | Should -Be $paths.ClocExe
            }

            It "runs cloc script with perl when cloc.exe not found" {
                $paths = @{}
                $paths.Root = "C\\repo"
                $paths.ClocExe = "C\\tools\cloc.exe"
                $paths.ClocScript = "C\\tools\cloc"

                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
                Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith {
                    [pscustomobject]@{ Path = "C:\perl\bin\perl.exe" }
                }

                $global:LASTEXITCODE = 0
                $runner = {
                    param($Command, $Arguments)
                    $script:executedCommand = $Command
                    $script:executedArgs = $Arguments
                }

                { Invoke-ClocCount -Paths $paths -IsWindows $false -InvokeProcess $runner } | Should -Not -Throw
                $script:executedCommand | Should -Be "C:\perl\bin\perl.exe"
                $script:executedArgs | Should -Contain $paths.ClocScript
            }

            It "throws when perl is not found for cloc script" {
                $paths = @{}
                $paths.Root = "C\\repo"
                $paths.ClocExe = "C\\tools\cloc.exe"
                $paths.ClocScript = "C\\tools\cloc"

                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
                Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith { $null }

                { Invoke-ClocCount -Paths $paths -IsWindows $false } | Should -Throw "Perl is required to run the bundled cloc script."
            }

            It "throws when no cloc binary found" {
                $paths = @{}
                $paths.Root = "C\\repo"
                $paths.ClocExe = "C\\tools\cloc.exe"
                $paths.ClocScript = "C\\tools\cloc"

                Mock -CommandName Test-Path -MockWith { $false }

                { Invoke-ClocCount -Paths $paths -IsWindows $true } | Should -Throw "Bundled cloc binary not found."
            }

            It "prefers cloc.exe on Windows even when cloc script exists" {
                $paths = @{}
                $paths.Root = "C\\repo"
                $paths.ClocExe = "C\\tools\cloc.exe"
                $paths.ClocScript = "C\\tools\cloc"

                Mock -CommandName Test-Path -MockWith { $true }
                $global:LASTEXITCODE = 0
                $script:whichPath = $null

                $runner = {
                    param($Command, $Arguments)
                    $script:whichPath = $Command
                    $null = $Arguments
                }

                Invoke-ClocCount -Paths $paths -IsWindows $true -InvokeProcess $runner
                $script:whichPath | Should -Be $paths.ClocExe
            }

            It "uses cloc script on non-Windows platforms" {
                $paths = @{}
                $paths.Root = "/home/repo"
                $paths.ClocExe = "/home/tools/cloc.exe"
                $paths.ClocScript = "/home/tools/cloc"

                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
                Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
                Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith {
                    [pscustomobject]@{ Path = "/usr/bin/perl" }
                }

                $global:LASTEXITCODE = 0
                $runner = {
                    param($Command, $Arguments)
                    $script:executedCommand = $Command
                    $script:executedArgs = $Arguments
                }

                { Invoke-ClocCount -Paths $paths -IsWindows $false -InvokeProcess $runner } | Should -Not -Throw
                $script:executedCommand | Should -Be "/usr/bin/perl"
                $script:executedArgs | Should -Contain $paths.ClocScript
            }
        }

        Context "Integration scenarios" {
            It "throws when no bundled cloc is found" {
                Mock -CommandName Resolve-Path -MockWith { param($Path) $null = $Path; [pscustomobject]@{ Path = "C:\repo" } }
                Mock -CommandName Test-Path -MockWith { $false }

                { & $script:scriptPath -Path "C:\repo" } | Should -Throw "Bundled cloc binary not found."
            }

            It "accepts custom path parameter" {
                Mock -CommandName Resolve-Path -MockWith {
                    param($Path)
                    [pscustomobject]@{ Path = $Path }
                }
                Mock -CommandName Test-Path -MockWith { $false }

                { & $script:scriptPath -Path "C:\custom\path" } | Should -Throw
            }
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

        # Import only the functions we need (gold-standard pattern)
        . (Import-ScriptFunction -Path $scriptPath -Name "Invoke-LPassExe")
        . (Import-ScriptFunction -Path $scriptPath -Name "Get-LPassSecret")
        . (Import-ScriptFunction -Path $scriptPath -Name "Invoke-LoadOpenAIKey")

        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "lpass" } -MockWith {
            [pscustomobject]@{ Name = "lpass" }
        }

        # Mock the wrapper (NOT the native executable)
        Mock -CommandName Invoke-LPassExe -MockWith {
            param([string[]]$Args)

            # Ensure script sees success
            $global:LASTEXITCODE = 0

            # Optional strictness: validate expected call shape
            # ($Args -join ' ') | Should -BeExactly 'show Test Item --notes'

            return "secret-value"
        }

        Mock -CommandName Set-Item -MockWith {
            param($Path, $Value)
            $setCalls.Add(@{ Path = $Path; Value = $Value })
        }

        # Call the function entrypoint (preferred for unit tests)
        Invoke-LoadOpenAIKey -ItemName "Test Item" -EnvVar "TEST_ENV"

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

}





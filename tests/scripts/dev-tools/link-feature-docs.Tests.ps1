Set-StrictMode -Version Latest

Describe "link-feature-docs.ps1" {
    BeforeAll {
        $env:POSHQC_SKIP_SCRIPT_EXECUTION = '1'
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\link-feature-docs.ps1"
        . $script:scriptPath
        function global:gh {
            param([Parameter(ValueFromRemainingArguments = $true)]$Args)
            $null = $Args
        }
    }

    Context "Build-FeatureDocumentationBlock function" {
        It "creates docs block with feature name in paths" {
            $block = Build-FeatureDocumentationBlock -FeatureName "my-feature"
            $block | Should -Match "## Feature Docs"
            $block | Should -Match "docs/features/active/my-feature/user-story.md"
            $block | Should -Match "docs/features/active/my-feature/spec.md"
            $block | Should -Match "docs/features/active/my-feature/plan.md"
        }

        It "handles feature names with underscores" {
            $block = Build-FeatureDocumentationBlock -FeatureName "my_feature_name"
            $block | Should -Match "docs/features/active/my_feature_name/user-story.md"
            $block | Should -Match "docs/features/active/my_feature_name/spec.md"
            $block | Should -Match "docs/features/active/my_feature_name/plan.md"
        }

        It "handles feature names with hyphens" {
            $block = Build-FeatureDocumentationBlock -FeatureName "my-feature-name"
            $block | Should -Match "docs/features/active/my-feature-name/user-story.md"
            $block | Should -Match "docs/features/active/my-feature-name/spec.md"
            $block | Should -Match "docs/features/active/my-feature-name/plan.md"
        }

        It "creates properly formatted markdown links" {
            $block = Build-FeatureDocumentationBlock -FeatureName "test"
            $block | Should -Match "\[User Story\]\(docs/features/active/test/user-story.md\)"
            $block | Should -Match "\[Spec\]\(docs/features/active/test/spec.md\)"
            $block | Should -Match "\[Plan\]\(docs/features/active/test/plan.md\)"
        }
    }

    Context "Set-OrAppendSection function" {
        It "replaces existing section content" {
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
            $content = "## Intro`nhello"
            $replacement = "## Feature Docs`ncontent"
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement $replacement
            $pattern = ([regex]::Escape($replacement.TrimEnd())) -replace "\\n", "\r?\n"
            $updated | Should -Match $pattern
        }

        It "handles empty content by returning replacement" {
            $updated = Set-OrAppendSection -Content "" -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`ncontent"
            $updated | Should -Match "## Feature Docs"
            $updated | Should -Match "content"
        }

        It "handles whitespace-only content by returning replacement" {
            $updated = Set-OrAppendSection -Content "   `n  `n  " -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`ncontent"
            $updated | Should -Match "## Feature Docs"
            $updated | Should -Match "content"
        }

        It "replaces section at start of content" {
            $content = @"
## Feature Docs
old content

## Next Section
other
"@
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew"
            $updated | Should -Match "new"
            $updated | Should -Not -Match "old content"
            $updated | Should -Match "## Next Section"
        }

        It "replaces section in middle of content" {
            $content = @"
## First
first content

## Feature Docs
old content

## Last
last content
"@
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew"
            $updated | Should -Match "## First"
            $updated | Should -Match "new"
            $updated | Should -Not -Match "old content"
            $updated | Should -Match "## Last"
        }

        It "replaces section at end of content" {
            $content = @"
## First
first content

## Feature Docs
old content
"@
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew"
            $updated | Should -Match "## First"
            $updated | Should -Match "new"
            $updated | Should -Not -Match "old content"
        }

        It "trims trailing whitespace from replacement" {
            $content = "## Intro`nhello"
            $replacement = "## Feature Docs`ncontent   `n`n`n"
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement $replacement
            $updated | Should -Not -Match "content\s+$"
        }

        It "supports ShouldProcess and returns original content when WhatIf" {
            $content = @"
## Intro
hello

## Feature Docs
old body
"@
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew body" -WhatIf
            $updated | Should -Be $content
        }

        It "handles sections with special regex characters" {
            $content = @"
## Test (Special)
old content

## Next
other
"@
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Test (Special)" -Replacement "## Test (Special)`nnew"
            $updated | Should -Match "new"
            $updated | Should -Not -Match "old content"
        }

        It "handles content with CRLF line endings" {
            $content = "## Intro`r`nhello`r`n`r`n## Feature Docs`r`nold body"
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`r`nnew body"
            $updated | Should -Match "new body"
            $updated | Should -Not -Match "old body"
        }

        It "handles content with Unix line endings" {
            $content = "## Intro`nhello`n`n## Feature Docs`nold body"
            $updated = Set-OrAppendSection -Content $content -SectionHeading "## Feature Docs" -Replacement "## Feature Docs`nnew body"
            $updated | Should -Match "new body"
            $updated | Should -Not -Match "old body"
        }
    }

    Context "Invoke-LinkFeatureDocument" {
        BeforeEach {
            $script:ghCalls = New-Object System.Collections.Generic.List[object]
            Mock -CommandName Get-Command -ParameterFilter { $Name -eq 'gh' } -MockWith { @{ Name = 'gh' } }
            Mock -CommandName Invoke-GhCli -MockWith {
                param([string[]]$GhArgs)
                $script:ghCalls.Add($GhArgs) | Out-Null
                $global:LASTEXITCODE = 0

                if ($GhArgs[0] -eq 'issue' -and $GhArgs[1] -eq 'view') {
                    return @{ Output = '{"body":"## Intro`nexisting"}'; ExitCode = 0 }
                }

                if ($GhArgs[0] -eq 'issue' -and $GhArgs[1] -eq 'edit') {
                    return @{ Output = ""; ExitCode = 0 }
                }

                return @{ Output = ""; ExitCode = 0 }
            }
            Mock -CommandName Set-Content -MockWith { param($Path, $Value, $Encoding) $script:lastWrite = @{ Path = $Path; Value = $Value; Encoding = $Encoding } }
            Mock -CommandName Remove-Item -MockWith { }
            Mock -CommandName Write-Output -MockWith {
                param([Parameter(ValueFromRemainingArguments = $true)]$Message, $InputObject)
                if ($PSBoundParameters.ContainsKey('InputObject')) {
                    $script:lastMessage = $InputObject
                } else {
                    $script:lastMessage = $Message
                }
            }
        }

        It "throws when issue number is missing" {
            $action = { Invoke-LinkFeatureDocument -IssueNumberParam '' -FeatureNameParam 'feature' }
            Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Issue number is required*'
        }

        It "throws when feature name is missing" {
            $action = { Invoke-LinkFeatureDocument -IssueNumberParam '42' -FeatureNameParam '' }
            Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Feature name is required*'
        }

        It "throws when gh view fails" {
            Mock -CommandName Invoke-GhCli -MockWith {
                param([string[]]$GhArgs)
                [void] $GhArgs
                $global:LASTEXITCODE = 1
                return @{ Output = ""; ExitCode = 1 }
            }

            $action = { Invoke-LinkFeatureDocument -IssueNumberParam '42' -FeatureNameParam 'demo' }
            Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Unable to fetch issue #42*'
        }

        It "throws when issue body is empty" {
            Mock -CommandName Invoke-GhCli -MockWith {
                param([string[]]$GhArgs)
                $global:LASTEXITCODE = 0
                if ($GhArgs[0] -eq 'issue' -and $GhArgs[1] -eq 'view') { return @{ Output = '{"body":""}'; ExitCode = 0 } }
                return @{ Output = ""; ExitCode = 0 }
            }

            $action = { Invoke-LinkFeatureDocument -IssueNumberParam '42' -FeatureNameParam 'demo' }
            Should -ActualValue $action -Throw -ExceptionType ([System.InvalidOperationException]) -ExpectedMessage 'Issue #42 has an empty body*'
        }

        It "writes updated body and commits changes" {
            Invoke-LinkFeatureDocument -IssueNumberParam '42' -FeatureNameParam 'demo'

            $script:lastWrite.Value | Should -Match "## Feature Docs"
            $script:lastWrite.Path | Should -Not -BeNullOrEmpty
            $script:lastMessage | Should -Match 'Updated issue #42'
        }
    }
}

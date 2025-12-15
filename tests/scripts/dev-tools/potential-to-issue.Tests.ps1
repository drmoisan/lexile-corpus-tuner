Set-StrictMode -Version Latest

Describe "potential-to-issue.ps1" {
    BeforeAll {
        $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
        . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../powershell/Support/TestHelpers.ps1"))
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\potential-to-issue.ps1"
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
            $scriptContent = Get-Content -Path $script:scriptPath -Raw
            $scriptContent | Should -Match "function Write-ScriptError"
            $scriptContent | Should -Match "\[CmdletBinding\(\)\]"
            $scriptContent | Should -Match "param\(\s+\[Parameter\(Mandatory = \`$true\)\]\s+\[string\] \`$Message"
        }

        It "calls Write-Error in the function body" {
            $scriptContent = Get-Content -Path $script:scriptPath -Raw
            $scriptContent | Should -Match "function Write-ScriptError[\s\S]+?Write-Error.*Message"
        }

        It "calls exit 1 in the function body" {
            $scriptContent = Get-Content -Path $script:scriptPath -Raw
            $scriptContent | Should -Match "function Write-ScriptError[\s\S]+?exit 1"
        }
    }
}

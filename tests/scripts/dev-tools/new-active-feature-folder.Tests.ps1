Set-StrictMode -Version Latest

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
. (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../../powershell/Support/TestHelpers.ps1"))

Describe "new-active-feature-folder.ps1 helpers" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\new-active-feature-folder.ps1"
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

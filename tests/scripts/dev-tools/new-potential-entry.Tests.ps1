Set-StrictMode -Version Latest

BeforeAll {
    $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
    . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../powershell/Support/TestHelpers.ps1"))
}

Describe "new-potential-entry.ps1 - Test-ValidShortName" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
    }

    Context "Valid short names" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Test-ValidShortName")
        }

        It "accepts single lowercase word" {
            Test-ValidShortName -Name "feature" | Should -Be $true
        }

        It "accepts kebab-case with two words" {
            Test-ValidShortName -Name "my-feature" | Should -Be $true
        }

        It "accepts kebab-case with multiple words" {
            Test-ValidShortName -Name "my-new-feature" | Should -Be $true
        }

        It "accepts numbers in the name" {
            Test-ValidShortName -Name "feature-v2" | Should -Be $true
        }

        It "accepts name with only numbers" {
            Test-ValidShortName -Name "123" | Should -Be $true
        }

        It "accepts mixed alphanumeric kebab-case" {
            Test-ValidShortName -Name "abc123-def456" | Should -Be $true
        }
    }

    Context "Invalid short names" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Test-ValidShortName")
        }

        It "rejects uppercase letters" {
            Test-ValidShortName -Name "MyFeature" | Should -Be $false
        }

        It "rejects spaces" {
            Test-ValidShortName -Name "my feature" | Should -Be $false
        }

        It "rejects underscores" {
            Test-ValidShortName -Name "my_feature" | Should -Be $false
        }

        It "rejects trailing hyphen" {
            Test-ValidShortName -Name "feature-" | Should -Be $false
        }

        It "rejects leading hyphen" {
            Test-ValidShortName -Name "-feature" | Should -Be $false
        }

        It "rejects double hyphens" {
            Test-ValidShortName -Name "my--feature" | Should -Be $false
        }

        It "rejects special characters" {
            Test-ValidShortName -Name "my@feature" | Should -Be $false
        }

        It "rejects empty string" {
            Test-ValidShortName -Name "" | Should -Be $false
        }
    }
}

Describe "new-potential-entry.ps1 - Get-AuthorName" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
    }

    Context "Author name retrieval" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-AuthorName")
        }

        It "returns git config user.name when available" {
            $result = Get-AuthorName -GetGitConfig { param($Key) $null = $Key; "John Doe" }
            $result | Should -Be "John Doe"
        }

        It "falls back to USERNAME environment variable when git fails" {
            $result = Get-AuthorName `
                -GetGitConfig { param($Key) $null = $Key; $null } `
                -GetEnvironmentVariable { param($Name) $null = $Name; "WindowsUser" }
            $result | Should -Be "WindowsUser"
        }

        It "returns 'Unknown' when git returns empty and no USERNAME" {
            $result = Get-AuthorName `
                -GetGitConfig { param($Key) $null = $Key; "" } `
                -GetEnvironmentVariable { param($Name) $null = $Name; $null }
            $result | Should -Be "Unknown"
        }

        It "returns 'Unknown' when git returns whitespace only" {
            $result = Get-AuthorName `
                -GetGitConfig { param($Key) $null = $Key; "   " } `
                -GetEnvironmentVariable { param($Name) $null = $Name; $null }
            $result | Should -Be "Unknown"
        }
    }
}

Describe "new-potential-entry.ps1 - Convert-TemplateContent" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
    }

    Context "Template content replacement" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Convert-TemplateContent")
        }

        It "replaces feature-name placeholder" {
            $content = "# <feature-name> (Potential)"
            $result = Convert-TemplateContent -Content $content -ShortName "my-feature" -Date "2025-12-10" -Author "Test User"
            $result | Should -Be "# my-feature (Potential)"
        }

        It "replaces date placeholder" {
            $content = "- Date captured: YYYY-MM-DD"
            $result = Convert-TemplateContent -Content $content -ShortName "my-feature" -Date "2025-12-10" -Author "Test User"
            $result | Should -Be "- Date captured: 2025-12-10"
        }

        It "replaces author placeholder" {
            $content = "- Author: name"
            $result = Convert-TemplateContent -Content $content -ShortName "my-feature" -Date "2025-12-10" -Author "Test User"
            $result | Should -Be "- Author: Test User"
        }

        It "replaces all placeholders in complete template" {
            $content = @"
# <feature-name> (Potential)

- Date captured: YYYY-MM-DD
- Author: name
- Status: Draft
"@
            $result = Convert-TemplateContent -Content $content -ShortName "my-feature" -Date "2025-12-10" -Author "Test User"
            $result | Should -Match "# my-feature \(Potential\)"
            $result | Should -Match "- Date captured: 2025-12-10"
            $result | Should -Match "- Author: Test User"
            $result | Should -Not -Match "<feature-name>"
            $result | Should -Not -Match "YYYY-MM-DD"
            $result | Should -Not -Match "- Author: name"
        }

        It "handles multiple occurrences of feature-name" {
            $content = "<feature-name> and <feature-name>"
            $result = Convert-TemplateContent -Content $content -ShortName "test" -Date "2025-12-10" -Author "Test User"
            $result | Should -Be "test and test"
        }
    }
}

Describe "new-potential-entry.ps1 - Invoke-VSCodeOpen" {
    BeforeAll {
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
    }

    Context "VS Code command detection and execution" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-VSCodeOpen")
        }

        It "returns true when code command is available" {
            $result = Invoke-VSCodeOpen -Files @("file1.md", "file2.md") `
                -GetCommand { param($Name) $null = $Name; [pscustomobject]@{ Name = "code" } } `
                -StartProcess { param($FilePath, $ArgumentList) $null = $FilePath; $null = $ArgumentList }

            $result | Should -Be $true
        }

        It "returns false when code command is not available" {
            $result = Invoke-VSCodeOpen -Files @("file1.md", "file2.md") `
                -GetCommand { param($Name) $null = $Name; $null }

            $result | Should -Be $false
        }

        It "calls Start-Process with correct parameters" {
            $files = @("file1.md", "file2.md")
            $result = Invoke-VSCodeOpen -Files $files `
                -GetCommand { param($Name) $null = $Name; [pscustomobject]@{ Name = "code" } } `
                -StartProcess { param($FilePath, $ArgumentList) $FilePath | Should -Be 'code'; $ArgumentList | Should -Be $files }

            $result | Should -Be $true
        }
    }
}

Describe "new-potential-entry.ps1 - Integration validation" {
    Context "Script structure validation" {
        It "contains all expected function definitions" {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
            $scriptContent = Get-Content -Path $scriptPath -Raw

            $scriptContent | Should -Match "function Test-ValidShortName"
            $scriptContent | Should -Match "function Get-AuthorName"
            $scriptContent | Should -Match "function Convert-TemplateContent"
            $scriptContent | Should -Match "function Invoke-VSCodeOpen"
        }

        It "validates parameter declaration" {
            $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "../../../scripts/dev-tools/new-potential-entry.ps1"
            $scriptContent = Get-Content -Path $scriptPath -Raw

            $scriptContent | Should -Match "param\(\s*\[string\]\s*\`$ShortName\s*\)"
        }
    }
}

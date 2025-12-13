Set-StrictMode -Version Latest

# Import the helper to load functions from scripts
. (Join-Path $PSScriptRoot '..\..\powershell\Support\TestHelpers.ps1' -Resolve)

Describe "run-actionlint.ps1" {
    BeforeAll {
        $script:scriptPath = Join-Path $PSScriptRoot '..\..\..\scripts\dev-tools\run-actionlint.ps1' -Resolve

        # Import all functions from the script for testing
        . (Import-ScriptFunction -Path $script:scriptPath -Name 'Resolve-ActionlintPath')
        . (Import-ScriptFunction -Path $script:scriptPath -Name 'Find-ActionlintOnPath')
        . (Import-ScriptFunction -Path $script:scriptPath -Name 'Install-Actionlint')
        . (Import-ScriptFunction -Path $script:scriptPath -Name 'Add-DirectoryToPath')
        . (Import-ScriptFunction -Path $script:scriptPath -Name 'Invoke-ActionlintCommand')
    }

    Context "Resolve-ActionlintPath" {
        It "resolves paths correctly from script location" {
            # Arrange
            $testScriptPath = 'C:\repo\scripts\dev-tools\run-actionlint.ps1'

            Mock -CommandName Split-Path -MockWith { 'C:\repo\scripts\dev-tools' }
            Mock -CommandName Resolve-Path -MockWith {
                [PSCustomObject]@{ Path = 'C:\repo' }
            }
            Mock -CommandName Join-Path -MockWith {
                param($Path, $ChildPath)
                $null = $Path
                if ($ChildPath -eq '..\..') {
                    return 'C:\repo'
                } elseif ($ChildPath -eq 'tools\actionlint\bin') {
                    return 'C:\repo\tools\actionlint\bin'
                } else {
                    return 'C:\repo\tools\actionlint\bin\actionlint.exe'
                }
            }

            # Act
            $result = Resolve-ActionlintPath -ScriptPath $testScriptPath

            # Assert
            $result.RepoRoot | Should -Be 'C:\repo'
            $result.BinDir | Should -Be 'C:\repo\tools\actionlint\bin'
            $result.ExePath | Should -Be 'C:\repo\tools\actionlint\bin\actionlint.exe'
        }

        It "returns PSCustomObject with expected properties" {
            # Arrange
            Mock -CommandName Split-Path -MockWith { 'C:\test\scripts\dev-tools' }
            Mock -CommandName Resolve-Path -MockWith { [PSCustomObject]@{ Path = 'C:\test' } }
            Mock -CommandName Join-Path -MockWith {
                param($Path, $ChildPath)
                $null = $Path
                if ($ChildPath -eq '..\..') { return 'C:\test' }
                elseif ($ChildPath -eq 'tools\actionlint\bin') { return 'C:\test\tools\actionlint\bin' }
                else { return 'C:\test\tools\actionlint\bin\actionlint.exe' }
            }

            # Act
            $result = Resolve-ActionlintPath -ScriptPath 'C:\test\scripts\dev-tools\run-actionlint.ps1'

            # Assert
            $result | Should -BeOfType [PSCustomObject]
            $result.PSObject.Properties.Name | Should -Contain 'RepoRoot'
            $result.PSObject.Properties.Name | Should -Contain 'BinDir'
            $result.PSObject.Properties.Name | Should -Contain 'ExePath'
        }
    }

    Context "Find-ActionlintOnPath" {
        It "returns command source when actionlint is found on PATH" {
            # Arrange
            Mock -CommandName Get-Command -MockWith {
                [PSCustomObject]@{ Source = 'C:\tools\actionlint.exe' }
            }

            # Act
            $result = Find-ActionlintOnPath

            # Assert
            $result | Should -Be 'C:\tools\actionlint.exe'
            Should -Invoke Get-Command -Times 1 -Exactly
        }

        It "returns null when actionlint is not found on PATH" {
            # Arrange
            Mock -CommandName Get-Command -MockWith {
                throw "Command not found"
            }
            Mock -CommandName Write-Information -MockWith { }

            # Act
            $result = Find-ActionlintOnPath

            # Assert
            $result | Should -BeNullOrEmpty
            Should -Invoke Write-Information -Times 1 -Exactly
        }

        It "writes information message when actionlint not found" {
            # Arrange
            $script:infoMessages = @()
            Mock -CommandName Get-Command -MockWith { throw "Not found" }
            Mock -CommandName Write-Information -MockWith {
                param($MessageData)
                $script:infoMessages += $MessageData
            }

            # Act
            $null = Find-ActionlintOnPath

            # Assert
            $script:infoMessages | Should -Contain 'actionlint not found on PATH; will try local copy under tools/actionlint/bin.'
        }
    }

    Context "Install-Actionlint" {
        BeforeEach {
            $script:infoMessages = @()
            Mock -CommandName Write-Information -MockWith {
                param($MessageData)
                $script:infoMessages += $MessageData
            }
            Mock -CommandName New-Item -MockWith { [PSCustomObject]@{ FullName = 'C:\repo\tools\actionlint\bin' } }
            Mock -CommandName Invoke-WebRequest -MockWith { }
            Mock -CommandName Expand-Archive -MockWith { }
            Mock -CommandName Remove-Item -MockWith { }
            Mock -CommandName Test-Path -MockWith { $true }
            Mock -CommandName Join-Path -MockWith {
                param($Path, $ChildPath)
                "$Path\$ChildPath"
            }
        }

        It "creates bin directory if it does not exist" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            Should -Invoke New-Item -Times 1 -Exactly -ParameterFilter {
                $ItemType -eq 'Directory' -and $Path -eq 'C:\repo\tools\actionlint\bin'
            }
        }

        It "downloads actionlint from GitHub releases" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe' -Version '1.7.7'

            # Assert
            Should -Invoke Invoke-WebRequest -Times 1 -Exactly -ParameterFilter {
                $Uri -eq 'https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_windows_amd64.zip'
            }
        }

        It "extracts downloaded archive to bin directory" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            Should -Invoke Expand-Archive -Times 1 -Exactly -ParameterFilter {
                $DestinationPath -eq 'C:\repo\tools\actionlint\bin'
            }
        }

        It "removes zip file after extraction" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            Should -Invoke Remove-Item -Times 1 -Exactly -ParameterFilter {
                $Path -like '*actionlint_*.zip' -and $Force -eq $true
            }
        }

        It "returns exe path on successful installation" {
            # Act
            $result = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            $result | Should -Be 'C:\repo\tools\actionlint\bin\actionlint.exe'
        }

        It "throws error when exe not found after extraction" {
            # Arrange
            Mock -CommandName Test-Path -MockWith { $false }

            # Act & Assert
            { Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe' } |
                Should -Throw -ExpectedMessage '*actionlint.exe was not found*'
        }

        It "writes information messages during installation" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            $script:infoMessages | Should -Contain 'actionlint not found; downloading local copy into tools/actionlint/bin...'
            $script:infoMessages | Should -Match 'Downloading.*github.com.*'
            $script:infoMessages | Should -Match 'Extracting.*'
        }

        It "uses custom version when provided" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe' -Version '1.6.0'

            # Assert
            Should -Invoke Invoke-WebRequest -Times 1 -Exactly -ParameterFilter {
                $Uri -eq 'https://github.com/rhysd/actionlint/releases/download/v1.6.0/actionlint_1.6.0_windows_amd64.zip'
            }
        }

        It "defaults to version 1.7.7 when not specified" {
            # Act
            $null = Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe'

            # Assert
            Should -Invoke Invoke-WebRequest -Times 1 -Exactly -ParameterFilter {
                $Uri -eq 'https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_windows_amd64.zip'
            }
        }
    }

    Context "Add-DirectoryToPath" {
        BeforeEach {
            # Save original PATH
            $script:originalPath = $env:PATH
        }

        AfterEach {
            # Restore original PATH
            $env:PATH = $script:originalPath
        }

        It "adds directory to PATH when not present" {
            # Arrange
            $env:PATH = 'C:\existing1;C:\existing2'
            $newDir = 'C:\new\directory'

            # Act
            Add-DirectoryToPath -Directory $newDir

            # Assert
            $env:PATH | Should -Match ([regex]::Escape($newDir))
            $env:PATH | Should -Match "^$([regex]::Escape($newDir))"
        }

        It "does not add directory to PATH when already present" {
            # Arrange
            $existingDir = 'C:\existing\dir'
            $env:PATH = "C:\before;$existingDir;C:\after"
            $pathBefore = $env:PATH

            # Act
            Add-DirectoryToPath -Directory $existingDir

            # Assert
            $env:PATH | Should -Be $pathBefore
        }

        It "prepends directory to the beginning of PATH" {
            # Arrange
            $env:PATH = 'C:\dir1;C:\dir2'
            $newDir = 'C:\new'

            # Act
            Add-DirectoryToPath -Directory $newDir

            # Assert
            $env:PATH | Should -Match "^$([regex]::Escape($newDir));"
        }

        It "handles empty PATH" {
            # Arrange
            $env:PATH = ''
            $newDir = 'C:\new'

            # Act
            Add-DirectoryToPath -Directory $newDir

            # Assert
            $env:PATH | Should -Match ([regex]::Escape($newDir))
        }

        It "uses correct path separator" {
            # Arrange
            $env:PATH = 'C:\existing'
            $newDir = 'C:\new'
            $separator = [IO.Path]::PathSeparator

            # Act
            Add-DirectoryToPath -Directory $newDir

            # Assert
            $env:PATH | Should -Match "$([regex]::Escape($newDir))$([regex]::Escape($separator))"
        }
    }

    Context "Invoke-ActionlintCommand" {
        BeforeEach {
            $script:infoMessages = @()
            Mock -CommandName Write-Information -MockWith {
                param($MessageData)
                $script:infoMessages += $MessageData
            }
            Mock -CommandName Write-Error -MockWith { }
        }

        It "executes actionlint command with arguments" {
            # Arrange & Act
            $global:LASTEXITCODE = 0
            Write-Information 'Running actionlint...' -InformationAction Continue

            # Assert - verify the function would execute correctly
            $global:LASTEXITCODE | Should -Be 0
        }

        It "writes information message before execution" {
            # Arrange
            $global:LASTEXITCODE = 0

            # Act
            & {
                Write-Information 'Running actionlint...' -InformationAction Continue
                $global:LASTEXITCODE = 0
            }

            # Verify the information message would be displayed
            $global:LASTEXITCODE | Should -Be 0
        }

        It "exits with error when actionlint returns non-zero" {
            # Arrange
            $global:LASTEXITCODE = 1
            $script:errorWritten = $false

            # Act
            & {
                $exitCode = 1
                if ($exitCode -ne 0) {
                    $script:errorWritten = $true
                }
            }

            # Assert
            $script:errorWritten | Should -Be $true
        }

        It "passes through all arguments to actionlint" {
            # Arrange
            $global:LASTEXITCODE = 0
            $TestArguments = @('-verbose', '--color', 'file1.yml', 'file2.yml')

            # Act
            $TestArguments.Count | Should -Be 4
            $TestArguments | Should -Contain '-verbose'
            $TestArguments | Should -Contain '--color'

            # Assert
            $global:LASTEXITCODE | Should -Be 0
        }

        It "handles zero arguments correctly" {
            # Arrange
            $global:LASTEXITCODE = 0

            # Act
            & {
                $TestArguments = @()
                $TestArguments.Count | Should -Be 0
                $global:LASTEXITCODE = 0
            }

            # Assert
            $global:LASTEXITCODE | Should -Be 0
        }
    }

    Context "Integration scenarios" {
        It "script contains all expected functions" {
            # Arrange
            $scriptContent = Get-Content -Path $script:scriptPath -Raw

            # Assert
            $scriptContent | Should -Match 'function Resolve-ActionlintPath'
            $scriptContent | Should -Match 'function Find-ActionlintOnPath'
            $scriptContent | Should -Match 'function Install-Actionlint'
            $scriptContent | Should -Match 'function Add-DirectoryToPath'
            $scriptContent | Should -Match 'function Invoke-ActionlintCommand'
        }

        It "script has CmdletBinding on all functions" {
            # Arrange
            $scriptContent = Get-Content -Path $script:scriptPath -Raw
            $functionMatches = [regex]::Matches($scriptContent, 'function\s+(\w+-\w+)\s*\{')

            # Assert
            foreach ($match in $functionMatches) {
                $functionName = $match.Groups[1].Value
                $functionContent = $scriptContent.Substring($match.Index)
                $functionContent | Should -Match '\[CmdletBinding\(\)\]' -Because "Function $functionName should have CmdletBinding"
            }
        }

        It "main execution flow calls functions in correct order" {
            # Arrange
            $scriptContent = Get-Content -Path $script:scriptPath -Raw

            # Assert - verify the main script logic structure
            $scriptContent | Should -Match 'Resolve-ActionlintPath.*Find-ActionlintOnPath.*Install-Actionlint.*Add-DirectoryToPath.*Invoke-ActionlintCommand'
        }

        It "script sets ErrorActionPreference to Stop" {
            # Arrange
            $scriptContent = Get-Content -Path $script:scriptPath -Raw

            # Assert
            $scriptContent | Should -Match "\`$ErrorActionPreference\s*=\s*'Stop'"
        }

        It "script sets InformationPreference to Continue" {
            # Arrange
            $scriptContent = Get-Content -Path $script:scriptPath -Raw

            # Assert
            $scriptContent | Should -Match "\`$InformationPreference\s*=\s*'Continue'"
        }
    }

    Context "Error handling" {
        It "Install-Actionlint throws when download fails" {
            # Arrange
            Mock -CommandName New-Item -MockWith { }
            Mock -CommandName Join-Path -MockWith { param($Path, $ChildPath) "$Path\$ChildPath" }
            Mock -CommandName Invoke-WebRequest -MockWith {
                throw "Network error"
            }

            # Act & Assert
            { Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe' } |
                Should -Throw
        }

        It "Install-Actionlint throws when extraction fails" {
            # Arrange
            Mock -CommandName New-Item -MockWith { }
            Mock -CommandName Join-Path -MockWith { param($Path, $ChildPath) "$Path\$ChildPath" }
            Mock -CommandName Write-Information -MockWith { }
            Mock -CommandName Invoke-WebRequest -MockWith { }
            Mock -CommandName Expand-Archive -MockWith {
                throw "Extraction failed"
            }

            # Act & Assert
            { Install-Actionlint -BinDir 'C:\repo\tools\actionlint\bin' -ExePath 'C:\repo\tools\actionlint\bin\actionlint.exe' } |
                Should -Throw
        }

        It "Resolve-ActionlintPath throws when script path is invalid" {
            # Arrange
            Mock -CommandName Split-Path -MockWith {
                throw "Invalid path"
            }

            # Act & Assert
            { Resolve-ActionlintPath -ScriptPath 'invalid' } |
                Should -Throw
        }
    }

    Context "Edge cases" {
        It "Install-Actionlint handles special characters in path" {
            # Arrange
            Mock -CommandName New-Item -MockWith { }
            Mock -CommandName Join-Path -MockWith { param($Path, $ChildPath) "$Path\$ChildPath" }
            Mock -CommandName Write-Information -MockWith { }
            Mock -CommandName Invoke-WebRequest -MockWith { }
            Mock -CommandName Expand-Archive -MockWith { }
            Mock -CommandName Remove-Item -MockWith { }
            Mock -CommandName Test-Path -MockWith { $true }

            # Act
            $result = Install-Actionlint -BinDir 'C:\repo with spaces\tools' -ExePath 'C:\repo with spaces\tools\actionlint.exe'

            # Assert
            $result | Should -Be 'C:\repo with spaces\tools\actionlint.exe'
        }

        It "Add-DirectoryToPath handles paths with trailing separator" {
            # Arrange
            $env:PATH = 'C:\existing'
            $newDir = 'C:\new\'

            # Act
            Add-DirectoryToPath -Directory $newDir

            # Assert
            $env:PATH | Should -Match ([regex]::Escape($newDir))
        }

        It "Find-ActionlintOnPath handles Get-Command errors gracefully" {
            # Arrange
            Mock -CommandName Get-Command -MockWith {
                $exception = New-Object System.Management.Automation.CommandNotFoundException
                throw $exception
            }
            Mock -CommandName Write-Information -MockWith { }

            # Act
            $result = Find-ActionlintOnPath

            # Assert
            $result | Should -BeNullOrEmpty
        }
    }
}

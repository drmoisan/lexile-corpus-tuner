[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAlignAssignmentStatement', '', Justification = 'Test mock data structures with nested hash tables are difficult to align uniformly')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', '', Justification = 'Mock function parameters mirror real function signatures for testing')]
param()

Set-StrictMode -Version Latest

BeforeAll {
    $modulePath = Join-Path $PSScriptRoot '../../scripts/powershell/PoshQC/PoshQC.psm1'
    Import-Module -Name $modulePath -Force
    $moduleInfo = Get-Module PoshQC
    $moduleRoot = Split-Path -Parent $moduleInfo.Path
    $script:TestSettingsPath = Join-Path $moduleRoot 'settings/pssa.settings.psd1'
}

Describe 'Get-PoshQCFileList' {
    Context 'When given a valid root path' {
        It 'Should resolve the root path and return PowerShell files' {
            InModuleScope PoshQC {
                $testRoot = $PSScriptRoot
                Mock -CommandName Resolve-Path -MockWith { [PSCustomObject]@{ Path = $testRoot } }
                Mock -CommandName Get-ChildItem -MockWith {
                    @(
                        [PSCustomObject]@{ FullName = "$testRoot/test.ps1"; Extension = '.ps1' },
                        [PSCustomObject]@{ FullName = "$testRoot/module.psm1"; Extension = '.psm1' }
                    )
                }

                $result = Get-PoshQCFileList -Root $testRoot

                $result.Count | Should -Be 2
                Should -Invoke -CommandName Resolve-Path -Times 1 -Exactly
                Should -Invoke -CommandName Get-ChildItem -Times 1 -Exactly
            }
        }

        It 'Should exclude files in excluded directories' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                Mock -CommandName Resolve-Path -MockWith { [PSCustomObject]@{ Path = $testRoot } }
                Mock -CommandName Get-ChildItem -MockWith {
                    @(
                        [PSCustomObject]@{ FullName = "$testRoot/scripts/test.ps1"; Extension = '.ps1' },
                        [PSCustomObject]@{ FullName = "$testRoot/.git/hooks.ps1"; Extension = '.ps1' },
                        [PSCustomObject]@{ FullName = "$testRoot/node_modules/lib.ps1"; Extension = '.ps1' }
                    )
                }

                $result = Get-PoshQCFileList -Root $testRoot

                $result.Count | Should -Be 1
                $result[0].FullName | Should -Be "$testRoot/scripts/test.ps1"
            }
        }

        It 'Should return empty array when no files found' {
            InModuleScope PoshQC {
                $testRoot = '/empty'
                Mock -CommandName Resolve-Path -MockWith { [PSCustomObject]@{ Path = $testRoot } }
                Mock -CommandName Get-ChildItem -MockWith { $null }

                $result = Get-PoshQCFileList -Root $testRoot

                $result | Should -BeNullOrEmpty
            }
        }

        It 'Should accept custom exclusion list' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                Mock -CommandName Resolve-Path -MockWith { [PSCustomObject]@{ Path = $testRoot } }
                Mock -CommandName Get-ChildItem -MockWith {
                    @(
                        [PSCustomObject]@{ FullName = "$testRoot/scripts/test.ps1"; Extension = '.ps1' },
                        [PSCustomObject]@{ FullName = "$testRoot/custom/excluded.ps1"; Extension = '.ps1' }
                    )
                }

                $result = Get-PoshQCFileList -Root $testRoot -ExcludeDirs @('custom')

                $result.Count | Should -Be 1
                $result[0].FullName | Should -Be "$testRoot/scripts/test.ps1"
            }
        }
    }

    Context 'When given an invalid root path' {
        It 'Should throw an error when path cannot be resolved' {
            InModuleScope PoshQC {
                Mock -CommandName Resolve-Path -MockWith { throw 'Path not found' }

                { Get-PoshQCFileList -Root 'C:\nonexistent' } | Should -Throw
            }
        }
    }
}

Describe 'Install-PoshQCTool' {
    Context 'When PSGallery is not registered' {
        It 'Should register PSGallery when not found' -Skip {
            # Skip: Mocking Register-PSRepository is problematic due to dynamic parameters
        }
    }

    Context 'When PSGallery is not trusted' {
        It 'Should attempt to set PSGallery as trusted' -Skip {
            # Skip: Mocking Set-PSRepository is problematic
        }

        It 'Should handle failure to set PSGallery as trusted gracefully' -Skip {
            # Skip: Mocking Set-PSRepository is problematic
        }
    }

    Context 'When modules need to be installed' {
        It 'Should install missing modules' -Skip {
            # Skip: Mocking Install-Module and complex Get-Module behavior is brittle
        }

        It 'Should verify module installation after install' -Skip {
            # Skip: Mocking Install-Module and complex Get-Module behavior is brittle
        }

        It 'Should throw when module installation fails' -Skip {
            # Skip: Mocking Install-Module and complex Get-Module behavior is brittle
        }
    }

    Context 'When modules are already installed' {
        It 'Should skip installation for already-present modules' -Skip {
            # Skip: Covered by integration test in PoshQC.EntryPoints.Tests.ps1
        }
    }
}

Describe 'Invoke-PoshQCFormat' {
    Context 'When PSScriptAnalyzer is not installed' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { $null }

                { Invoke-PoshQCFormat -Root $PSScriptRoot } | Should -Throw '*PSScriptAnalyzer is not installed*'
            }
        }
    }

    Context 'When settings file does not exist' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $false }

                { Invoke-PoshQCFormat -Root $PSScriptRoot -SettingsPath 'C:\nonexistent.psd1' } | Should -Throw '*Settings not found*'
            }
        }
    }

    Context 'When formatting files' {
        It 'Should handle files that do not need formatting' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1" })
                }
                Mock -CommandName Get-Content -MockWith { "Write-Host 'test'" }
                Mock -CommandName Invoke-Formatter -MockWith { "Write-Host 'test'" }
                Mock -CommandName Set-Content -MockWith { }

                { Invoke-PoshQCFormat -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Set-Content -Times 0 -Exactly
            }
        }

        It 'Should format files that need formatting' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1" })
                }
                Mock -CommandName Get-Content -MockWith { "Write-Host  'test'" }
                Mock -CommandName Invoke-Formatter -MockWith { "Write-Host 'test'" }
                Mock -CommandName Set-Content -MockWith { }

                { Invoke-PoshQCFormat -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Set-Content -Times 1 -Exactly
            }
        }

        It 'Should normalize line endings before formatting' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1" })
                }
                Mock -CommandName Get-Content -MockWith { "Write-Host 'test'`r`n" }
                Mock -CommandName Invoke-Formatter -MockWith {
                    param($ScriptDefinition)
                    $ScriptDefinition | Should -Not -Match "`r`n"
                    return "Write-Host 'test'`n"
                }
                Mock -CommandName Set-Content -MockWith { }

                { Invoke-PoshQCFormat -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw
            }
        }
    }
}

Describe 'Invoke-PoshQCAnalyze' {
    Context 'When PSScriptAnalyzer is not installed' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { $null }

                { Invoke-PoshQCAnalyze -Root $PSScriptRoot } | Should -Throw '*PSScriptAnalyzer is not installed*'
            }
        }
    }

    Context 'When settings file does not exist' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $false }

                { Invoke-PoshQCAnalyze -Root $PSScriptRoot -SettingsPath 'C:\nonexistent.psd1' } | Should -Throw '*Settings not found*'
            }
        }
    }

    Context 'When analyzing files' {
        It 'Should pass when no issues are found' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1"; Extension = '.ps1' })
                }
                Mock -CommandName Invoke-ScriptAnalyzer -MockWith { @() }

                { Invoke-PoshQCAnalyze -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw
            }
        }

        It 'Should throw when issues are found' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1"; Extension = '.ps1' })
                }
                Mock -CommandName Invoke-ScriptAnalyzer -MockWith {
                    @([PSCustomObject]@{ Message = 'Test issue'; Severity = 'Warning' })
                }
                Mock -CommandName Format-Table -MockWith { }

                { Invoke-PoshQCAnalyze -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Throw '*reported 1 issue*'
            }
        }

        It 'Should filter files by extension (.ps1, .psm1)' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @(
                        [PSCustomObject]@{ FullName = "$testRoot/test.ps1"; Extension = '.ps1' },
                        [PSCustomObject]@{ FullName = "$testRoot/module.psm1"; Extension = '.psm1' },
                        [PSCustomObject]@{ FullName = "$testRoot/data.psd1"; Extension = '.psd1' }
                    )
                }
                Mock -CommandName Invoke-ScriptAnalyzer -MockWith { @() }

                { Invoke-PoshQCAnalyze -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Invoke-ScriptAnalyzer -Times 2 -Exactly
            }
        }

        It 'Should handle Invoke-ScriptAnalyzer failures with detailed error' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'PSScriptAnalyzer' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Get-PoshQCFileList -MockWith {
                    @([PSCustomObject]@{ FullName = "$testRoot/test.ps1"; Extension = '.ps1' })
                }
                Mock -CommandName Invoke-ScriptAnalyzer -MockWith {
                    throw [System.InvalidOperationException]::new('Parse error')
                }

                { Invoke-PoshQCAnalyze -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Throw '*Invoke-ScriptAnalyzer failed*'
            }
        }
    }
}

Describe 'Invoke-PoshQCTest' {
    Context 'When Pester is not installed' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { $null }

                { Invoke-PoshQCTest -Root $PSScriptRoot } | Should -Throw '*Pester is not installed*'
            }
        }
    }

    Context 'When settings file does not exist' {
        It 'Should throw an error' {
            InModuleScope PoshQC {
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $false }

                { Invoke-PoshQCTest -Root $PSScriptRoot -SettingsPath 'C:\nonexistent.psd1' } | Should -Throw '*Settings not found*'
            }
        }
    }

    Context 'When running tests' {
        It 'Should handle no test files gracefully' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{ Enabled = $false }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = @("$testRoot/tests") }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = $null
                    }
                    return $config
                }
                Mock -CommandName Test-Path -MockWith {
                    param($Path)
                    # Settings path check
                    if ($Path -eq $testSettings) {
                        return $true
                    }
                    # Run path check
                    if ($Path -eq "$testRoot/tests") {
                        return $true
                    }
                    return $false
                }
                Mock -CommandName Get-ChildItem -MockWith { @() }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw
            }
        }

        It 'Should resolve paths relative to root' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests', 'scripts') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{ Enabled = $false }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = $Hashtable.Run.Path }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = $null
                    }
                    return $config
                }
                Mock -CommandName Get-ChildItem -MockWith { @() }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw
            }
        }

        It 'Should apply ExcludeDirs to paths' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests', '.venv', 'scripts') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{ Enabled = $false }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = $Hashtable.Run.Path }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = $null
                    }
                    return $config
                }
                Mock -CommandName Get-ChildItem -MockWith { @() }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -ExcludeDirs @('.venv') -InformationAction SilentlyContinue } | Should -Not -Throw
            }
        }

        It 'Should create result directory when needed' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{
                            Enabled = $true
                            OutputPath = 'artifacts/results.xml'
                            OutputFormat = 'JUnitXml'
                        }
                        CodeCoverage = @{ Enabled = $false }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = $Hashtable.Run.Path }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $true }
                            OutputPath = @{ Value = $Hashtable.TestResult.OutputPath }
                        }
                        CodeCoverage = $null
                    }
                    return $config
                }
                Mock -CommandName New-Item -MockWith { }
                Mock -CommandName Get-ChildItem -MockWith { @() }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName New-Item -Times 1 -Exactly
            }
        }
    }

    Context 'When coverage is enabled' {
        It 'Should resolve coverage paths' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{
                            Enabled = $true
                            Path = @('src/**/*.ps1')
                            OutputPath = 'artifacts/coverage.xml'
                            OutputFormat = 'CoverageGutters'
                        }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = $Hashtable.Run.Path }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = @{
                            Enabled = @{ Value = $true }
                            Path = @{ Value = $Hashtable.CodeCoverage.Path }
                            OutputPath = @{ Value = $Hashtable.CodeCoverage.OutputPath }
                        }
                    }
                    return $config
                }
                Mock -CommandName New-Item -MockWith { }
                Mock -CommandName Get-ChildItem -MockWith { @() }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName New-Item -Times 1 -Exactly
            }
        }

        It 'Should generate Koverage copy by default when coverage is enabled' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{
                            Enabled = $true
                            Path = @('src/**/*.ps1')
                            OutputPath = 'artifacts/coverage.xml'
                            OutputFormat = 'CoverageGutters'
                        }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = @("$testRoot/tests") }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = @{
                            Enabled = @{ Value = $true }
                            Path = @{ Value = $Hashtable.CodeCoverage.Path }
                            OutputPath = @{ Value = "$testRoot/artifacts/coverage.xml" }
                        }
                    }
                    return $config
                }
                Mock -CommandName New-Item -MockWith { }
                Mock -CommandName Test-Path -MockWith {
                    param($Path)
                    # Settings path check
                    if ($Path -eq $testSettings) {
                        return $true
                    }
                    # Run path check
                    if ($Path -eq "$testRoot/tests") {
                        return $true
                    }
                    return $false
                }
                Mock -CommandName Invoke-Pester -MockWith { }
                Mock -CommandName Convert-PoshQCCoverageToRelative -MockWith { }

                $enumerateTestsStub = {
                    param([string[]] $Paths, [string[]] $Excluded, [scriptblock] $TestPathFn)
                    @([PSCustomObject]@{ FullName = "$testRoot/tests/test.Tests.ps1" })
                }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -EnumerateTests $enumerateTestsStub -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Invoke-Pester -Times 1 -Exactly
                Should -Invoke -CommandName Convert-PoshQCCoverageToRelative -Times 1 -Exactly
            }
        }

        It 'Should skip Koverage copy when DisableKoverageCopy is set' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Test-Path -MockWith { $true }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{
                            Enabled = $true
                            Path = @('src/**/*.ps1')
                            OutputPath = 'artifacts/coverage.xml'
                            OutputFormat = 'CoverageGutters'
                        }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = $Hashtable.Run.Path }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = @{
                            Enabled = @{ Value = $true }
                            Path = @{ Value = $Hashtable.CodeCoverage.Path }
                            OutputPath = @{ Value = 'artifacts/coverage.xml' }
                        }
                    }
                    return $config
                }
                Mock -CommandName New-Item -MockWith { }
                Mock -CommandName Invoke-Pester -MockWith { }
                Mock -CommandName Convert-PoshQCCoverageToRelative -MockWith { }

                $enumerateTestsStub = {
                    param([string[]] $Paths, [string[]] $Excluded, [scriptblock] $TestPathFn)
                    @([PSCustomObject]@{ FullName = "$testRoot/tests/test.Tests.ps1" })
                }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -EnumerateTests $enumerateTestsStub -DisableKoverageCopy -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Invoke-Pester -Times 1 -Exactly
                Should -Invoke -CommandName Convert-PoshQCCoverageToRelative -Times 0 -Exactly
            }
        }

        It 'Should use custom KoverageOutputPath when provided' {
            InModuleScope PoshQC {
                $testRoot = '/repo'
                $testSettings = '/settings.psd1'
                $customKoveragePath = '/custom/koverage.xml'
                Mock -CommandName Get-Module -MockWith { [PSCustomObject]@{ Name = 'Pester' } }
                Mock -CommandName Import-Module -MockWith { }
                Mock -CommandName Import-PowerShellDataFile -MockWith {
                    @{
                        Run = @{ Path = @('tests') }
                        Should = @{ ErrorAction = 'Stop' }
                        Output = @{ Verbosity = 'Detailed' }
                        TestResult = @{ Enabled = $false }
                        CodeCoverage = @{
                            Enabled = $true
                            Path = @('src/**/*.ps1')
                            OutputPath = 'artifacts/coverage.xml'
                            OutputFormat = 'CoverageGutters'
                        }
                    }
                }
                Mock -CommandName New-PesterConfiguration -MockWith {
                    param($Hashtable)
                    $config = [PSCustomObject]@{
                        Run = @{
                            Path = @{ Value = @("$testRoot/tests") }
                            ExcludePath = @{ Value = @() }
                        }
                        TestResult = @{
                            Enabled = @{ Value = $false }
                            OutputPath = @{ Value = $null }
                        }
                        CodeCoverage = @{
                            Enabled = @{ Value = $true }
                            Path = @{ Value = $Hashtable.CodeCoverage.Path }
                            OutputPath = @{ Value = "$testRoot/artifacts/coverage.xml" }
                        }
                    }
                    return $config
                }
                Mock -CommandName New-Item -MockWith { }
                Mock -CommandName Test-Path -MockWith {
                    param($Path)
                    # Settings path check
                    if ($Path -eq $testSettings) {
                        return $true
                    }
                    # Run path check
                    if ($Path -eq "$testRoot/tests") {
                        return $true
                    }
                    return $false
                }
                Mock -CommandName Invoke-Pester -MockWith { }
                Mock -CommandName Convert-PoshQCCoverageToRelative -MockWith { }

                $enumerateTestsStub = {
                    param([string[]] $Paths, [string[]] $Excluded, [scriptblock] $TestPathFn)
                    @([PSCustomObject]@{ FullName = "$testRoot/tests/test.Tests.ps1" })
                }

                { Invoke-PoshQCTest -Root $testRoot -SettingsPath $testSettings -EnumerateTests $enumerateTestsStub -KoverageOutputPath $customKoveragePath -InformationAction SilentlyContinue } | Should -Not -Throw

                Should -Invoke -CommandName Invoke-Pester -Times 1 -Exactly
                Should -Invoke -CommandName Convert-PoshQCCoverageToRelative -ParameterFilter { $OutputPath -eq $customKoveragePath } -Times 1 -Exactly
            }
        }
    }
}


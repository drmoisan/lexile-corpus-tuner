Set-StrictMode -Version Latest

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '../../scripts/powershell/PoshQC/PoshQC.psm1') -Force
}

Describe 'Get-PoshQCFileList' {
    It 'returns empty when no PowerShell files are enumerated' {
        $resolvePath = {
            param($Path)
            'resolvePath-called' | Out-Null
            $Path
        }
        $enumerateFiles = {
            param($Path)
            'enumerateFiles-called' | Out-Null
            [void] $Path
            @()
        }

        $result = Get-PoshQCFileList -Root '/repo' -ResolvePath $resolvePath -EnumerateFiles $enumerateFiles

        $result | Should -BeNullOrEmpty
    }

    It 'excludes files matched by the injected predicate' {
        $resolvePath = { param($Path) $Path }
        $enumerateFiles = {
            param($Path)
            [void] $Path
            @([pscustomobject]@{ FullName = '/repo/skip/file.ps1'; Extension = '.ps1' })
        }
        $shouldExclude = {
            param($File, [string[]] $ExcludedDirs)
            [void] $ExcludedDirs
            $File.FullName -match '/skip/'
        }

        $result = Get-PoshQCFileList -Root '/repo' -ExcludeDirs @('skip') -ResolvePath $resolvePath -EnumerateFiles $enumerateFiles -ShouldExclude $shouldExclude

        $result | Should -BeNullOrEmpty
    }

    It 'throws when the injected resolver fails' {
        { Get-PoshQCFileList -Root '/missing' -ResolvePath { throw 'not found' } } |
            Should -Throw "Failed to resolve root path '/missing': not found"
    }

    It 'filters out non-PowerShell extensions with the injected predicate' {
        $result = Get-PoshQCFileList -Root '/repo' -ResolvePath { param([string] $Path) $Path } -EnumerateFiles {
            param([string] $Path)
            [void] $Path
            @(
                [pscustomobject]@{ FullName = '/repo/a.ps1'; Extension = '.ps1' },
                [pscustomobject]@{ FullName = '/repo/b.txt'; Extension = '.txt' }
            )
        } -IsAllowedExtension {
            param($File)
            $File.Extension -eq '.ps1'
        }

        $result | Should -HaveCount 1
        $result[0].FullName | Should -Be '/repo/a.ps1'
    }

    It 'resolves relative roots using the injected resolver' {
        $received = $null
        $resolvePath = { param([string] $Path) [void] $Path; '/abs/scripts' }
        $enumerateFiles = {
            param([string] $Path)
            Set-Variable -Scope 1 -Name 'received' -Value $Path
            @([pscustomobject]@{ FullName = (Join-Path $Path 'tool.psm1'); Extension = '.psm1' })
        }

        $result = Get-PoshQCFileList -Root 'scripts' -ResolvePath $resolvePath -EnumerateFiles $enumerateFiles

        ($received -replace '\\', '/') | Should -Be '/abs/scripts'
        $result | Should -HaveCount 1
        ($result[0].FullName -replace '\\', '/') | Should -Be '/abs/scripts/tool.psm1'
    }
}

Describe 'Install-PoshQCTool' {
    It 'registers PSGallery when absent and installs required modules' {
        $registered = $false
        $installed = @{}
        $logger = New-Object System.Collections.Generic.List[string]

        Install-PoshQCTool -GetRepository { $null } -RegisterRepository { Set-Variable -Scope 1 -Name 'registered' -Value $true } `
            -FindModule {
            param([string] $Name)
            [void] $Name
            if ($installed.ContainsKey($Name)) {
                [pscustomobject]@{ Name = $Name; Version = $installed[$Name] }
            }
        } -InstallModule {
            param([string] $Name, [string] $Version)
            [void] $Name
            [void] $Version
            $installed[$Name] = [version] $Version
        } -Logger {
            param([string] $Message, [string] $Level)
            [void] $Message
            [void] $Level
            $logger.Add("$($Level):$($Message)") | Out-Null
        } -SetTls { } -SetRepository { } | Out-Null

        $registered | Should -BeTrue
        ($installed.Keys | Sort-Object) | Should -Be @('Pester', 'PSScriptAnalyzer')
    }

    It 'logs a warning when PSGallery trust cannot be set' {
        $logs = New-Object System.Collections.Generic.List[string]

        Install-PoshQCTool -GetRepository { [pscustomobject]@{ InstallationPolicy = 'Untrusted' } } `
            -SetRepository { throw 'cannot set' } -FindModule {
            param([string] $Name)
            [pscustomobject]@{ Name = $Name; Version = [version]'9.9.9' }
        } `
            -InstallModule { param([string] $Name, [string] $Version) [void] $Name; [void] $Version } -RegisterRepository { } -SetTls { } -Logger {
            param([string] $Message, [string] $Level)
            [void] $Message
            [void] $Level
            $logs.Add("$($Level):$($Message)") | Out-Null
        } | Out-Null

        ($logs | Where-Object { $_ -like 'Warning:*PSGallery*' }).Count | Should -Be 1
    }

    It 'skips installation when required modules are already present' {
        $installCalls = 0

        Install-PoshQCTool -FindModule {
            param([string] $Name)
            [pscustomobject]@{ Name = $Name; Version = [version]'9.9.9' }
        } -InstallModule {
            param([string] $Name, [string] $Version)
            [void] $Name
            [void] $Version
            $script:installCalls++
        } -SetTls { } -GetRepository { [pscustomobject]@{ InstallationPolicy = 'Trusted' } } `
            -SetRepository { } -RegisterRepository { } -Logger {
            param([string] $Message, [string] $Level)
            [void] $Message
            [void] $Level
        } | Out-Null

        $installCalls | Should -Be 0
    }

    It 'throws when an injected install operation fails' {
        { Install-PoshQCTool -FindModule { param([string] $Name) [void] $Name } -InstallModule { param([string] $Name, [string] $Version) [void] $Name; [void] $Version; throw 'boom' } -SetTls { } -GetRepository { $null } -RegisterRepository { } -SetRepository { } -Logger { param([string] $Message, [string] $Level) [void] $Message; [void] $Level } } |
            Should -Throw '*Failed to install PSScriptAnalyzer 1.22.0: boom*'
    }
}

Describe 'Invoke-PoshQCFormat' {
    It 'throws when PSScriptAnalyzer is missing' {
        { Invoke-PoshQCFormat -EnsureModule { param([string] $Name, [string] $ErrorMessage) [void] $Name; throw $ErrorMessage } } |
            Should -Throw '*PSScriptAnalyzer is not installed*'
    }

    It 'throws when the settings file is missing' {
        { Invoke-PoshQCFormat -Root '/repo' -SettingsPath '/missing.psd1' -EnsureModule { } -TestPathExists { $false } } |
            Should -Throw 'Settings not found: /missing.psd1'
    }

    It 'logs and returns when no files are found' {
        $logs = New-Object System.Collections.Generic.List[string]

        Invoke-PoshQCFormat -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } `
            -GetFileList { @() } -Logger { param([string] $Message) $logs.Add($Message) | Out-Null } | Out-Null

        $logs | Should -Contain 'No PowerShell files found under /repo'
    }

    It 'writes formatted content when the formatter changes text' {
        $writes = @()

        Invoke-PoshQCFormat -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } `
            -GetFileList { param([string] $RootPath, [string[]] $Excluded) [void] $RootPath; [void] $Excluded; @([pscustomobject]@{ FullName = '/repo/file.ps1' }) } -ReadFile { param([string] $Path) [void] $Path; "Write-Host  `"test`"`r`n" } `
            -FormatContent {
            param([string] $Content, [string] $Settings)
            [void] $Settings
            $Content | Should -Not -Match "`r`n"
            "Write-Host 'test'`n"
        } -WriteFile {
            param([string] $Path, [string] $Content)
            Set-Variable -Scope 1 -Name 'writes' -Value ($writes + @(@{ Path = $Path; Content = $Content }))
        } -Logger { param([string] $Message) [void] $Message } | Out-Null

        $writes | Should -HaveCount 1
        $writes[0].Path | Should -Be '/repo/file.ps1'
        $writes[0].Content | Should -Be "Write-Host 'test'`n"
    }

    It 'performs no writes when content is unchanged' {
        $writes = 0

        Invoke-PoshQCFormat -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } `
            -GetFileList { @([pscustomobject]@{ FullName = '/repo/file.ps1' }) } -ReadFile { param([string] $Path) [void] $Path; "Write-Host 'same'`n" } `
            -FormatContent { param([string] $Content, [string] $Settings) [void] $Settings; $Content } -WriteFile {
            param([string] $Path, [string] $Content)
            [void] $Path
            [void] $Content
            $script:writes++
        } -Logger { param([string] $Message) [void] $Message } | Out-Null

        $writes | Should -Be 0
    }
}

Describe 'Invoke-PoshQCAnalyze' {
    It 'throws when PSScriptAnalyzer is missing' {
        { Invoke-PoshQCAnalyze -EnsureModule { param([string] $Name, [string] $ErrorMessage) [void] $Name; throw $ErrorMessage } } |
            Should -Throw '*PSScriptAnalyzer is not installed*'
    }

    It 'throws when the settings file is missing' {
        { Invoke-PoshQCAnalyze -Root '/repo' -SettingsPath '/missing.psd1' -EnsureModule { } -TestPathExists { $false } } |
            Should -Throw 'Settings not found: /missing.psd1'
    }

    It 'logs and returns when no files are found' {
        $logs = New-Object System.Collections.Generic.List[string]

        Invoke-PoshQCAnalyze -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } `
            -GetFileList { @() } -Logger { param([string] $Message) $logs.Add($Message) | Out-Null } -AnalyzeFile { param([string] $Path) [void] $Path; throw 'should not run' } | Out-Null

        $logs | Should -Contain 'No PowerShell files found under /repo'
    }

    It 'throws when analyzer returns findings' {
        { Invoke-PoshQCAnalyze -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } `
                -GetFileList { @([pscustomobject]@{ FullName = '/repo/file.ps1'; Extension = '.ps1' }) } `
                -AnalyzeFile { param([string] $Path) [void] $Path; @([pscustomobject]@{ Message = 'warn' }) } `
                -Logger { param([string] $Message) [void] $Message } } |
            Should -Throw 'PSScriptAnalyzer reported 1 issue(s).'
    }
}

Describe 'Convert-PoshQCCoverageToRelative' {
    Context 'When converting coverage paths with repo root' {
        It 'Should convert forward-slash paths in XML to relative paths' {
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="/tmp/repos/lexile-corpus-tuner/scripts/dev-tools">
  <class name="/tmp/repos/lexile-corpus-tuner/scripts/dev-tools/collect-commit-context" sourcefilename="collect-commit-context.ps1">
  </class>
  </package>
  <package name="/tmp/repos/lexile-corpus-tuner/scripts/powershell/PoshQC">
  <sourcefile name="PoshQC.psm1">
  </sourcefile>
  </package>
</report>
'@

            $result = Convert-PoshQCCoverageToRelative -InputContent $mockXmlContent -RepoRoot '/tmp/repos/lexile-corpus-tuner' -PassThru

            $result | Should -Not -Match '/tmp/repos/lexile-corpus-tuner/'
            $result | Should -Match '<package name="scripts/dev-tools">'
            $result | Should -Match '<class name="scripts/dev-tools/collect-commit-context"'
            $result | Should -Match '<package name="scripts/powershell/PoshQC">'
        }

        It 'Should convert backslash paths in XML to relative paths (Windows-style)' {
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="C:\repos\lexile-corpus-tuner\scripts\dev-tools">
  <class name="C:\repos\lexile-corpus-tuner\scripts\dev-tools\collect-commit-context" sourcefilename="collect-commit-context.ps1">
  </class>
  </package>
</report>
'@

            $result = Convert-PoshQCCoverageToRelative -InputContent $mockXmlContent -RepoRoot 'C:\repos\lexile-corpus-tuner' -PassThru

            $result | Should -Not -Match 'C:/repos/lexile-corpus-tuner/'
            $result | Should -Not -Match 'C:\\repos\\lexile-corpus-tuner\\'
            $result | Should -Match 'scripts'
        }

        It 'Should handle mixed forward and backslash paths' {
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="C:/repos/lexile-corpus-tuner/scripts/dev-tools">
  <class name="C:\repos\lexile-corpus-tuner\scripts\powershell\PoshQC" sourcefilename="PoshQC.psm1">
  </class>
  </package>
</report>
'@

            $result = Convert-PoshQCCoverageToRelative -InputContent $mockXmlContent -RepoRoot 'C:\repos\lexile-corpus-tuner' -PassThru

            $result | Should -Not -Match 'C:/repos/lexile-corpus-tuner/'
            $result | Should -Not -Match 'C:\\repos\\lexile-corpus-tuner\\'
            $result | Should -Match 'scripts'
        }
    }

    Context 'When RepoRoot has trailing separator' {
        It 'Should still convert paths correctly' {
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="/tmp/repos/lexile-corpus-tuner/scripts/dev-tools">
  </package>
</report>
'@

            $result = Convert-PoshQCCoverageToRelative -InputContent $mockXmlContent -RepoRoot '/tmp/repos/lexile-corpus-tuner/' -PassThru

            $result | Should -Not -Match '/tmp/repos/lexile-corpus-tuner/'
            $result | Should -Match '<package name="scripts/dev-tools">'
        }
    }

    Context 'When inputs or outputs are injected' {
        It 'skips conversion when neither InputPath nor InputContent is provided' {
            $logs = New-Object System.Collections.Generic.List[string]

            Convert-PoshQCCoverageToRelative -Logger { param([string] $Message) $logs.Add($Message) | Out-Null } | Out-Null

            $logs | Should -Contain 'No coverage input provided; skipping conversion.'
        }

        It 'derives default output path when only InputPath is supplied' {
            $writtenPath = $null
            $writtenContent = $null

            Convert-PoshQCCoverageToRelative -RepoRoot '/repo' -InputPath 'artifacts/pester/coverage.xml' -ResolvePath { param([string] $Path) if ($Path.StartsWith('/repo')) { $Path } else { "/repo/$Path" } } `
                -JoinPath { param([string] $Parent, [string] $Child) "$Parent/$Child" } -TestPathExists { param([string] $Path) [void] $Path; $true } -ReadContent { '<report></report>' } `
                -WriteContent {
                param([string] $Path, [string] $Content)
                Set-Variable -Scope 1 -Name 'writtenPath' -Value $Path
                Set-Variable -Scope 1 -Name 'writtenContent' -Value $Content
            } -EnsureDirectory { param([string] $Path) [void] $Path } -GetDefaultOutputPath {
                param([string] $ResolvedInputPath, [string] $ResolvedRoot)
                [void] $ResolvedInputPath
                "$ResolvedRoot/artifacts/pester/coverage.koverage.xml"
            } -Logger { param([string] $Message) [void] $Message } | Out-Null

            $writtenPath | Should -Be '/repo/artifacts/pester/coverage.koverage.xml'
            $writtenContent | Should -Be '<report></report>'
        }

        It 'returns converted content when PassThru is set with trailing RepoRoot separator' {
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="/tmp/repos/lexile-corpus-tuner/scripts/dev-tools">
  </package>
</report>
'@

            $result = Convert-PoshQCCoverageToRelative -InputContent $mockXmlContent -RepoRoot '/tmp/repos/lexile-corpus-tuner/' -PassThru

            $result | Should -Not -Match '/tmp/repos/lexile-corpus-tuner/'
            $result | Should -Match 'scripts/dev-tools'
        }
    }
}

Describe 'Invoke-PoshQCTest' {
    It 'throws when Pester is missing' {
        { Invoke-PoshQCTest -EnsureModule { param([string] $Name, [string] $ErrorMessage) [void] $Name; throw $ErrorMessage } } |
            Should -Throw '*Pester is not installed*'
    }

    It 'throws when settings file is missing' {
        { Invoke-PoshQCTest -Root '/repo' -SettingsPath '/missing.psd1' -EnsureModule { } -TestPathExists { $false } } |
            Should -Throw 'Settings not found: /missing.psd1'
    }

    It 'expands Run.Path and merges ExcludeDirs into ExcludePath' {
        $config = $null
        $script:capturedRunPaths = $null
        $script:capturedExcludes = $null
        $settings = @{
            Run          = @{ Path = @('tests') }
            Should       = @{ ErrorAction = 'Stop' }
            Output       = @{ Verbosity = 'Detailed' }
            TestResult   = @{ Enabled = $false }
            CodeCoverage = $null
        }

        Invoke-PoshQCTest -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } -LoadSettings { $settings } `
            -BuildConfiguration {
            param($Table)
            $config = [pscustomobject]@{
                Run          = @{
                    Path        = @{ Value = $Table.Run.Path }
                    ExcludePath = @{ Value = @() }
                }
                TestResult   = @{ Enabled = @{ Value = $false }; OutputPath = @{ Value = $null } }
                CodeCoverage = $null
                Output       = @{ Verbosity = 'Normal' }
            }
            return $config
        } -EnsureResultPath { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } -ExpandCoveragePaths { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } `
            -ExpandRunPaths {
            param($cfg, [string] $RootPath, [string[]] $Excluded)
            $cfg.Run.Path = @($cfg.Run.Path.Value | ForEach-Object { Join-Path $RootPath $_ })
            $cfg.Run.ExcludePath = @($Excluded | ForEach-Object { Join-Path $RootPath $_ })
            $cfg
        } `
            -EnumerateTests {
            param([string[]] $Paths, [string[]] $Excluded, [scriptblock] $TestPathFn)
            [void] $TestPathFn
            Set-Variable -Scope Script -Name capturedRunPaths -Value $Paths -Force
            Set-Variable -Scope Script -Name capturedExcludes -Value $Excluded -Force
            @([pscustomobject]@{ FullName = '/repo/tests/sample.Tests.ps1' })
        } -InvokePester {
            [pscustomobject]@{ Duration = [timespan]::Zero; PassedCount = 1; FailedCount = 0; SkippedCount = 0; InconclusiveCount = 0; NotRunCount = 0 }
        } -Logger { param([string] $Message) [void] $Message } -ExcludeDirs @('skip') | Out-Null

        ($script:capturedRunPaths | ForEach-Object { $_ -replace '\\', '/' }) | Should -Be @('/repo/tests')
        ($script:capturedExcludes | ForEach-Object { $_ -replace '\\', '/' }) | Should -Contain 'skip'
    }

    It 'logs and returns when no test files are found' {
        $logs = New-Object System.Collections.Generic.List[string]

        Invoke-PoshQCTest -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } -LoadSettings {
            @{ Run = @{ Path = @('tests') }; Should = @{ ErrorAction = 'Stop' }; Output = @{ Verbosity = 'Detailed' }; TestResult = @{ Enabled = $false }; CodeCoverage = $null }
        } -BuildConfiguration {
            param($Table)
            [pscustomobject]@{
                Run          = @{ Path = @{ Value = $Table.Run.Path }; ExcludePath = @{ Value = @() } }
                TestResult   = @{ Enabled = @{ Value = $false }; OutputPath = @{ Value = $null } }
                CodeCoverage = $null
                Output       = @{ Verbosity = 'Normal' }
            }
        } -EnsureResultPath { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } -ExpandCoveragePaths { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } `
            -EnumerateTests { @() } -Logger { param([string] $Message) $logs.Add($Message) | Out-Null } | Out-Null

        $logs | Should -Contain 'No Pester test files found under configured paths for root /repo'
    }

    It 'invokes coverage copy when coverage is enabled and not disabled' {
        $copyArgs = $null

        Invoke-PoshQCTest -Root '/repo' -SettingsPath '/settings.psd1' -EnsureModule { } -TestPathExists { $true } -LoadSettings {
            @{ Run = @{ Path = @('tests') }; Should = @{ ErrorAction = 'Stop' }; Output = @{ Verbosity = 'Detailed' }; TestResult = @{ Enabled = $false }; CodeCoverage = @{ Enabled = $true; Path = @('/repo/src'); OutputPath = '/repo/coverage.xml' } }
        } -BuildConfiguration {
            param($Table)
            [pscustomobject]@{
                Run          = @{ Path = @{ Value = $Table.Run.Path }; ExcludePath = @{ Value = @() } }
                TestResult   = @{ Enabled = @{ Value = $false }; OutputPath = @{ Value = $null } }
                CodeCoverage = @{ Enabled = $true; Path = @{ Value = $Table.CodeCoverage.Path }; OutputPath = @{ Value = $Table.CodeCoverage.OutputPath } }
                Output       = @{ Verbosity = 'Normal' }
            }
        } -EnsureResultPath { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } -ExpandCoveragePaths { param($cfg, [string] $RootPath) [void] $RootPath; $cfg } `
            -EnumerateTests { @([pscustomobject]@{ FullName = '/repo/tests/sample.Tests.ps1' }) } -InvokePester {
            [pscustomobject]@{
                Duration          = [timespan]::Zero
                PassedCount       = 1
                FailedCount       = 0
                SkippedCount      = 0
                InconclusiveCount = 0
                NotRunCount       = 0
                CodeCoverage      = [pscustomobject]@{ CoverageReport = 'Coverage: 100%' }
            }
        } -CopyCoverage {
            param([string] $CoveragePath, [string] $RepoRoot, [string] $KoveragePath)
            Set-Variable -Scope 1 -Name 'copyArgs' -Value @($CoveragePath, $RepoRoot, $KoveragePath)
        } -Logger { param([string] $Message) [void] $Message } | Out-Null

        $copyArgs[0] | Should -Be '/repo/coverage.xml'
        $copyArgs[1] | Should -Be '/repo'
        ($copyArgs[2] -replace '\\', '/') | Should -Be '/repo/coverage.koverage.xml'
    }
}


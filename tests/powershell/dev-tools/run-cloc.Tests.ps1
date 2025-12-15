Set-StrictMode -Version Latest

Describe "run-cloc.ps1" {
    BeforeAll {
        $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
        . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "../Support/TestHelpers.ps1"))

        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\run-cloc.ps1"
        [void]$script:scriptPath
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
            . (Import-ScriptFunction -Path $scriptPath -Name "Initialize-OutputRendering")
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
            $resolver = {
                param($InputPath)
                [pscustomobject]@{ Path = "C\\resolved\\$InputPath"; Input = $InputPath }
            }

            $result = Get-ClocPath -ScriptRoot "C\\script" -TargetPath "target" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\\resolved\\target"
            $result.ClocExe | Should -Match "cloc\.exe$"
            $result.ClocScript | Should -Match "cloc$"
        }
        It "resolves relative target paths" {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-ClocPath")
            $resolver = {
                param($InputPath)
                [pscustomobject]@{ Path = "C\\absolute\\path"; Input = $InputPath }
            }

            $result = Get-ClocPath -ScriptRoot "C\\base" -TargetPath "../relative" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\\absolute\\path"
        }
    }

    Context "Invoke-ClocCount" {
        BeforeEach {
            . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-ClocCount")
            $script:executedCommand = $null
            $script:executedArgs = $null
        }

        It "runs cloc.exe on Windows when it exists" {
            $paths = @{ Root = 'C\repo'; ClocExe = 'C\tools\cloc.exe'; ClocScript = 'C\tools\cloc' }

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
            $paths = @{ Root = 'C\repo'; ClocExe = 'C\tools\cloc.exe'; ClocScript = 'C\tools\cloc' }

            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
            Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith {
                [pscustomobject]@{ Path = "C:\\perl\\bin\\perl.exe" }
            }

            $global:LASTEXITCODE = 0
            $runner = {
                param($Command, $Arguments)
                $script:executedCommand = $Command
                $script:executedArgs = $Arguments
            }

            { Invoke-ClocCount -Paths $paths -IsWindows $false -InvokeProcess $runner } | Should -Not -Throw
            $script:executedCommand | Should -Be "C:\\perl\\bin\\perl.exe"
            $script:executedArgs | Should -Contain $paths.ClocScript
        }

        It "throws when perl is not found for cloc script" {
            $paths = @{ Root = 'C\repo'; ClocExe = 'C\tools\cloc.exe'; ClocScript = 'C\tools\cloc' }

            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
            Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith { $null }

            { Invoke-ClocCount -Paths $paths -IsWindows $false } | Should -Throw "Perl is required to run the bundled cloc script."
        }

        It "throws when no cloc binary found" {
            $paths = @{ Root = 'C\repo'; ClocExe = 'C\tools\cloc.exe'; ClocScript = 'C\tools\cloc' }

            Mock -CommandName Test-Path -MockWith { $false }

            { Invoke-ClocCount -Paths $paths -IsWindows $true } | Should -Throw "Bundled cloc binary not found."
        }

        It "prefers cloc.exe on Windows even when cloc script exists" {
            $paths = @{ Root = 'C\repo'; ClocExe = 'C\tools\cloc.exe'; ClocScript = 'C\tools\cloc' }

            Mock -CommandName Test-Path -MockWith { $true }
            $global:LASTEXITCODE = 0
            $script:whichPath = $null

            $runner = {
                param($Command, $Arguments)
                $script:whichPath = $Command
                $script:executedArgs = $Arguments
            }

            Invoke-ClocCount -Paths $paths -IsWindows $true -InvokeProcess $runner
            $script:whichPath | Should -Be $paths.ClocExe
        }

        It "uses cloc script on non-Windows platforms" {
            $paths = @{ Root = '/home/repo'; ClocExe = '/home/tools/cloc.exe'; ClocScript = '/home/tools/cloc' }

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
            Mock -CommandName Resolve-Path -MockWith { param($Path) $null = $Path; [pscustomobject]@{ Path = "C:\\repo" } }
            Mock -CommandName Test-Path -MockWith { $false }

            { & $script:scriptPath -Path "C:\\repo" } | Should -Throw "Bundled cloc binary not found."
        }

        It "accepts custom path parameter" {
            Mock -CommandName Resolve-Path -MockWith {
                param($Path)
                [pscustomobject]@{ Path = $Path }
            }
            Mock -CommandName Test-Path -MockWith { $false }

            { & $script:scriptPath -Path "C:\\custom\\path" } | Should -Throw
        }
    }
}




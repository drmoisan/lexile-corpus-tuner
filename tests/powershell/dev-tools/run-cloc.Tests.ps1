Set-StrictMode -Version Latest

function global:Import-ScriptFunction {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $resolved = (Resolve-Path -Path $Path).Path
    if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
        $null = $null
        $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($resolved, [ref]$null, [ref]$errors)
        if ($errors -and $errors.Count -gt 0) {
            throw "Failed to parse ${resolved}: $($errors[0].Message)"
        }

        $funcAst = $ast.Find(
            {
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                $node.Name -eq $Name
            },
            $true
        )

        if (-not $funcAst) {
            throw "Function $Name not found in $resolved"
        }

        return [scriptblock]::Create($funcAst.Extent.Text)
    }
}

Describe "run-cloc.ps1" {
    BeforeAll {
        $scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\scripts\dev-tools\run-cloc.ps1"
    }

    Context "Initialize-OutputRendering" {
        It "sets PSStyle.OutputRendering to PlainText on PowerShell 7+" {
            if ($PSVersionTable.PSVersion.Major -ge 7) {
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
            else {
                Set-ItResult -Skipped -Because "Test only applies to PowerShell 7+"
            }
        }

        It "does not error on Windows PowerShell 5.1" {
            . (Import-ScriptFunction -Path $scriptPath -Name "Initialize-OutputRendering")
            { Initialize-OutputRendering } | Should -Not -Throw
        }
    }

    Context "Test-IsWindows" {
        It "returns true on Windows PowerShell 5.1 when OS is Windows_NT" {
            . (Import-ScriptFunction -Path $scriptPath -Name "Test-IsWindows")
            if ($PSVersionTable.PSVersion.Major -lt 6) {
                $result = Test-IsWindows
                $result | Should -BeOfType [bool]
            }
            else {
                Set-ItResult -Skipped -Because "Test only applies to Windows PowerShell 5.1"
            }
        }

        It "returns platform detection on PowerShell 6+" {
            . (Import-ScriptFunction -Path $scriptPath -Name "Test-IsWindows")
            if ($PSVersionTable.PSVersion.Major -ge 6) {
                $result = Test-IsWindows
                $result | Should -Be $IsWindows
            }
            else {
                Set-ItResult -Skipped -Because "Test only applies to PowerShell 6+"
            }
        }
    }

    Context "Get-ClocPath" {
        It "constructs correct paths from script root and target path" {
            . (Import-ScriptFunction -Path $scriptPath -Name "Get-ClocPath")
            $resolver = { param($InputPath) [pscustomobject]@{ Path = "C\\resolved\\$InputPath" } }

            $result = Get-ClocPath -ScriptRoot "C\\script" -TargetPath "target" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\\resolved\\target"
            $result.ClocExe | Should -Match "tools\\cloc\\.exe$"
            $result.ClocScript | Should -Match "tools\\cloc$"
        }
        It "resolves relative target paths" {
            . (Import-ScriptFunction -Path $scriptPath -Name "Get-ClocPath")
            $resolver = {
                [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', '')]
                param($InputPath)
                [pscustomobject]@{ Path = "C\\absolute\\path" }
            }

            $result = Get-ClocPath -ScriptRoot "C\\base" -TargetPath "../relative" -ResolvePath $resolver -TestPath { $true }

            $result.Root | Should -Be "C\\absolute\\path"
        }
    Context "Invoke-ClocCount" {
        BeforeEach {
            . (Import-ScriptFunction -Path $scriptPath -Name "Invoke-ClocCount")
            $script:executedCommand = $null
            $script:executedArgs = $null
        }

        It "runs cloc.exe on Windows when it exists" {
            $paths = @{
                Root       = "C:\\repo"
                ClocExe    = "C:\\tools\cloc.exe"
                ClocScript = "C:\\tools\cloc"
            }

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
            $paths = @{
                Root       = "C:\\repo"
                ClocExe    = "C:\\tools\cloc.exe"
                ClocScript = "C:\\tools\cloc"
            }

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
            $paths = @{
                Root       = "C:\\repo"
                ClocExe    = "C:\\tools\cloc.exe"
                ClocScript = "C:\\tools\cloc"
            }

            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocExe } -MockWith { $false }
            Mock -CommandName Test-Path -ParameterFilter { $Path -eq $paths.ClocScript } -MockWith { $true }
            Mock -CommandName Get-Command -ParameterFilter { $Name -eq "perl" } -MockWith { $null }

            { Invoke-ClocCount -Paths $paths -IsWindows $false } | Should -Throw "Perl is required to run the bundled cloc script."
        }

        It "throws when no cloc binary found" {
            $paths = @{
                Root       = "C:\\repo"
                ClocExe    = "C:\\tools\cloc.exe"
                ClocScript = "C:\\tools\cloc"
            }

            Mock -CommandName Test-Path -MockWith { $false }

            { Invoke-ClocCount -Paths $paths -IsWindows $true } | Should -Throw "Bundled cloc binary not found."
        }

        It "prefers cloc.exe on Windows even when cloc script exists" {
            $paths = @{
                Root       = "C:\\repo"
                ClocExe    = "C:\\tools\cloc.exe"
                ClocScript = "C:\\tools\cloc"
            }

            Mock -CommandName Test-Path -MockWith { $true }
            $global:LASTEXITCODE = 0
            $script:whichPath = $null

            $runner = {
                param($Command, $Arguments)
                $script:whichPath = $Command
            }

            Invoke-ClocCount -Paths $paths -IsWindows $true -InvokeProcess $runner
            $script:whichPath | Should -Be $paths.ClocExe
        }

        It "uses cloc script on non-Windows platforms" {
            $paths = @{
                Root       = "/home/repo"
                ClocExe    = "/home/tools/cloc.exe"
                ClocScript = "/home/tools/cloc"
            }

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

            { & $scriptPath -Path "C:\\repo" } | Should -Throw "Bundled cloc binary not found."
        }

        It "accepts custom path parameter" {
            Mock -CommandName Resolve-Path -MockWith {
                param($Path)
                [pscustomobject]@{ Path = $Path }
            }
            Mock -CommandName Test-Path -MockWith { $false }

            { & $scriptPath -Path "C:\\custom\\path" } | Should -Throw
        }
    }
}
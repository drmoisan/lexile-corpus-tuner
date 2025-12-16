Set-StrictMode -Version Latest

Describe "load-openai-key.ps1" {
    BeforeAll {
        $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
        . (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath "..\..\powershell\Support\TestHelpers.ps1"))
        $script:scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\..\..\..\src\lexile_corpus_tuner\lexile_scoring_model\pipeline_scripts\load-openai-key.ps1"
    }

    It "sets environment variable when secret is returned" {
        $setCalls = New-Object System.Collections.Generic.List[hashtable]

        . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-LPassExe")
        . (Import-ScriptFunction -Path $script:scriptPath -Name "Get-LPassSecret")
        . (Import-ScriptFunction -Path $script:scriptPath -Name "Invoke-LoadOpenAIKey")

        Mock -CommandName Get-Command -ParameterFilter { $Name -eq "lpass" } -MockWith {
            [pscustomobject]@{ Name = "lpass" }
        }

        Mock -CommandName Invoke-LPassExe -MockWith {
            param([string[]]$LpassArgs)
            if ($LpassArgs) { $null = $LpassArgs.Count }
            $global:LASTEXITCODE = 0
            return "secret-value"
        }

        Mock -CommandName Set-Item -MockWith {
            param($Path, $Value)
            $setCalls.Add(@{ Path = $Path; Value = $Value })
        }

        Invoke-LoadOpenAIKey -ItemName "Test Item" -EnvVar "TEST_ENV"

        $setCalls.Count | Should -Be 1
        $setCalls[0].Path | Should -Be "Env:TEST_ENV"
        $setCalls[0].Value | Should -Be "secret-value"
    }
}

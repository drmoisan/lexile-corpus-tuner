Set-StrictMode -Version Latest
# PSScriptAnalyzerSuppressRule PSUseConsistentWhitespace
# PSScriptAnalyzerSuppressRule PSAlignAssignmentStatement

BeforeAll {
    $modulePath = Join-Path $PSScriptRoot '../../scripts/powershell/PoshQC/PoshQC.psm1'
    Import-Module -Name $modulePath -Force
    $moduleInfo = Get-Module PoshQC
    $moduleRoot = Split-Path -Parent $moduleInfo.Path
    $script:TestSettingsPath = Join-Path $moduleRoot 'settings/pssa.settings.psd1'
    if (-not (Test-Path -Path $script:TestSettingsPath)) {
        throw "Test settings file missing at $script:TestSettingsPath"
    }
}

# Note: These tests focus on behavioral integration rather than full isolation
# due to the tight coupling with PSScriptAnalyzer and Pester modules.
# Mocking core PowerShell cmdlets (Get-Module, Import-Module, etc.) is
# problematic and leads to brittle tests.

Describe 'Install-PoshQCTool' {
    It 'reports already-installed modules without error' {
        # This test verifies the function runs successfully when dependencies are present
        # It won't reinstall if PSScriptAnalyzer and Pester are already available
        { Install-PoshQCTool -InformationAction SilentlyContinue } | Should -Not -Throw
    }
}

Describe 'Invoke-PoshQCFormat' {
    It 'handles empty file list without error' {
        # Mock the internal Get-PoshQCFileList function within the module scope
        $testRoot = $PSScriptRoot
        $testSettings = $TestSettingsPath
        InModuleScope PoshQC -Parameters @{ testRoot = $testRoot; testSettings = $testSettings } {
            param($testRoot, $testSettings)
            # Suppress warnings about unused parameters (they ARE used in the Should assertion below)
            $null = $testRoot
            $null = $testSettings
            Mock -CommandName Get-PoshQCFileList -MockWith { @() }
            { Invoke-PoshQCFormat -Root $testRoot -SettingsPath $testSettings -InformationAction SilentlyContinue } | Should -Not -Throw
        }
    }
}

Describe 'Invoke-PoshQCAnalyze' {
    It 'validates settings path exists' {
        { Invoke-PoshQCAnalyze -Root $PSScriptRoot -SettingsPath 'C:\nonexistent\settings.psd1' } | Should -Throw '*Settings not found*'
    }
}

Describe 'Invoke-PoshQCTest' {
    It 'validates Pester module availability' {
        $pesterInstalled = Get-Module -ListAvailable -Name Pester
        $pesterInstalled | Should -Not -BeNullOrEmpty -Because 'Pester should be installed for these tests to run'
    }
}



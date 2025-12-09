@{
    RootModule           = 'PoshQC.psm1'
    ModuleVersion        = '0.1.1'
    GUID                 = '6a9b1a8b-1e8b-4c1f-9d6a-34b8e8c2f0e4'
    Author               = 'Dan Moisan'
    CompanyName          = 'Dan Moisan'
    Copyright            = '(c) Dan Moisan. All rights reserved.'
    Description          = 'Reusable PowerShell QC helpers (format, analyze, test) using Invoke-Formatter, PSScriptAnalyzer, and Pester.'
    PowerShellVersion    = '5.1'
    CompatiblePSEditions = @('Desktop', 'Core')
    RequiredModules      = @(
        @{ ModuleName = 'PSScriptAnalyzer'; ModuleVersion = '1.22.0' },
        @{ ModuleName = 'Pester'; ModuleVersion = '5.6.1' }
    )
    FunctionsToExport    = @(
        'Install-PoshQCTools',
        'Invoke-PoshQCFormat',
        'Invoke-PoshQCAnalyze',
        'Invoke-PoshQCTest'
    )
    CmdletsToExport      = @()
    VariablesToExport    = @()
    AliasesToExport      = @()
    PrivateData          = @{}
}


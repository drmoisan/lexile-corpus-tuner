@{
    Run        = @{
        Path = @('scripts', 'tests/powershell')
        Exit = $true
    }
    Should     = @{
        ErrorAction = 'Stop'
    }
    Output     = @{
        Verbosity = 'Detailed'
    }
    TestResult = @{
        Enabled      = $true
        OutputFormat = 'JUnitXml'
        OutputPath   = 'artifacts/pester/pester-junit.xml'
    }
}


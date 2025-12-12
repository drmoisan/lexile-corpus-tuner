@{
    Run          = @{
        Path = @('scripts', 'tests/powershell', 'tests/scripts')
        Exit = $true
    }

    Should       = @{
        ErrorAction = 'Stop'
    }

    Output       = @{
        Verbosity = 'Detailed'
    }

    TestResult   = @{
        Enabled      = $true
        OutputFormat = 'JUnitXml'
        OutputPath   = 'artifacts/pester/pester-junit.xml'
    }

    CodeCoverage = @{
        Enabled               = $true
        # Use Pester's CoverageGutters format so VS Code Coverage Gutters
        # can map file paths correctly.
        OutputFormat          = 'CoverageGutters'
        OutputPath            = 'artifacts/pester/powershell-coverage.xml'

        Path                  = @(
            'scripts/dev-tools/*.ps1'
            'scripts/powershell/**/*.psm1'
            'src/**/*.ps1'
        )

        # Optional: don't fail the run on coverage percentage
        CoveragePercentTarget = 0
    }
}




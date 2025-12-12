BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '../../scripts/powershell/PoshQC/PoshQC.psm1') -Force
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
            # Simulate Windows-style paths in XML on any platform
            $mockXmlContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<report name="Pester">
  <package name="C:\repos\lexile-corpus-tuner\scripts\dev-tools">
    <class name="C:\repos\lexile-corpus-tuner\scripts\dev-tools\collect-commit-context" sourcefilename="collect-commit-context.ps1">
    </class>
  </package>
</report>
'@

            # Pass InputContent directly to avoid path resolution; test regex matching only
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
}

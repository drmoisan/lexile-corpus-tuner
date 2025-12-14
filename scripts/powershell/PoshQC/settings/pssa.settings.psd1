@{
    # Analyze using all default rules and treat all severities as failures.
    IncludeDefaultRules = $true
    Severity            = @('Error', 'Warning', 'Information')

    Rules               = @{
        # Enforce compatibility with both Windows PowerShell 5.1 and PowerShell 7.4+
        PSUseCompatibleSyntax                          = @{
            Enable         = $true
            TargetVersions = @('5.1', '7.5')
            IgnoreUntested = $false
        }

        # Prefer 4-space indentation with pipeline indentation for clarity.
        PSUseConsistentIndentation                     = @{
            Enable              = $true
            IndentationSize     = 4
            PipelineIndentation = 'IncreaseIndentationAfterEveryPipeline'
            Kind                = 'space'
        }

        # Harden against dangerous patterns.
        PSAvoidUsingInvokeExpression                   = @{ Enable = $true }
        PSAvoidGlobalVars                              = @{ Enable = $true }
        PSAvoidUsingPlainTextForPassword               = @{ Enable = $true }
        PSAvoidUsingWriteHost                          = @{ Enable = $true }
        PSAvoidUsingConvertToSecureStringWithPlainText = @{ Enable = $true }

        # Require ShouldProcess for state-changing functions.
        PSUseShouldProcessForStateChangingFunctions    = @{
            Enable = $true
        }

        # Enforce consistent naming and indentation of assignments/whitespace.
        # PSAlignAssignmentStatement aligns equals signs by adding extra spaces.
        # PSUseConsistentWhitespace must allow this by NOT checking assignment operators.
        PSAlignAssignmentStatement                     = @{
            Enable         = $true
            CheckHashtable = $true
        }
        PSUseConsistentWhitespace                      = @{
            Enable                          = $true
            CheckInnerBrace                 = $true
            CheckOpenBrace                  = $true
            CheckOpenParen                  = $true
            CheckOperator                   = $false  # Allow aligned assignments
            CheckPipe                       = $true
            CheckPipeForRedundantWhitespace = $false
            CheckSeparator                  = $true
        }
    }
}




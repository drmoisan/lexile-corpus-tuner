<#
.SYNOPSIS
    Loads the OpenAI API key from a LastPass item into the current PowerShell session.

.DESCRIPTION
    Uses the LastPass CLI (`lpass`) to fetch either the secure note contents or the stored password
    for the specified item and assigns the value to OPENAI_API_KEY (or another env var).
    This keeps API keys out of tracked files while keeping the workflow quick to use.

.EXAMPLE
    ./src/lexile_corpus_tuner/pipeline_scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"

.EXAMPLE
    ./src/lexile_corpus_tuner/pipeline_scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key" -UsePasswordField -EnvVar "OPENAI_API_KEY"
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ItemName = "Lexile OpenAI Key",

    [Parameter()]
    [string]$EnvVar = "OPENAI_API_KEY",

    [Parameter()]
    [switch]$UsePasswordField,

    [Parameter()]
    [switch]$PrintOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSStyle.OutputRendering = 'PlainText'
}

function Invoke-LPassExe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string[]]$Args
    )

    # Capture stdout+stderr; tests will mock this function (not the native exe).
    lpass @Args 2>&1
}

function Get-LPassSecret {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ItemName,

        [Parameter(Mandatory)]
        [switch]$UsePasswordField
    )

    if (-not (Get-Command -Name lpass -ErrorAction SilentlyContinue)) {
        throw "LastPass CLI (lpass) is not installed or not in PATH."
    }

    $trimmedName = $ItemName.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmedName)) {
        throw "ItemName must not be empty."
    }

    $argsList = @('show', $trimmedName)
    if ($UsePasswordField) {
        $argsList += '--password'
    } else {
        $argsList += '--notes'
    }

    $output = Invoke-LPassExe -Args $argsList
    $code = $LASTEXITCODE

    # Normalize output to a single string; some hosts return an array of lines.
    $text =
        if ($null -eq $output) { '' }
        elseif ($output -is [System.Array]) { ($output -join "`n") }
        else { [string]$output }

    $text = $text.Trim()

    if ($code -ne 0 -or [string]::IsNullOrWhiteSpace($text)) {
        throw "Failed to fetch secret from LastPass item '$trimmedName'. Ensure you are logged in (run 'lpass login')."
    }

    return $text
}

function Invoke-LoadOpenAIKey {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$ItemName = "Lexile OpenAI Key",

        [Parameter()]
        [string]$EnvVar = "OPENAI_API_KEY",

        [Parameter()]
        [switch]$UsePasswordField,

        [Parameter()]
        [switch]$PrintOnly
    )

    $secret = Get-LPassSecret -ItemName $ItemName -UsePasswordField:$UsePasswordField

    if ($PrintOnly) {
        Write-Output $secret
        return
    }

    Set-Item -Path "Env:$EnvVar" -Value $secret
    Write-Output "Set $EnvVar for this session from LastPass item '$ItemName'."
}

# Script entrypoint: preserve the prior behavior (write error + exit 1) while keeping functions testable.
try {
    Invoke-LoadOpenAIKey -ItemName $ItemName -EnvVar $EnvVar -UsePasswordField:$UsePasswordField -PrintOnly:$PrintOnly
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}

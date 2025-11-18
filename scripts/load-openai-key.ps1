<#
.SYNOPSIS
    Loads the OpenAI API key from a LastPass item into the current PowerShell session.

.DESCRIPTION
    Uses the LastPass CLI (`lpass`) to fetch either the secure note contents or the stored password
    for the specified item and assigns the value to OPENAI_API_KEY (or another env var).
    This keeps API keys out of tracked files while keeping the workflow quick to use.

.EXAMPLE
    ./scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"

.EXAMPLE
    ./scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key" -UsePasswordField -EnvVar "OPENAI_API_KEY"
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

function Write-ErrorAndExit {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

if (-not (Get-Command -Name lpass -ErrorAction SilentlyContinue)) {
    Write-ErrorAndExit "LastPass CLI (lpass) is not installed or not in PATH."
}

$argsList = @("show", $ItemName.Trim())
if ($UsePasswordField) {
    $argsList += "--password"
} else {
    $argsList += "--notes"
}

$secret = & lpass @argsList 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($secret)) {
    Write-ErrorAndExit "Failed to fetch secret from LastPass item '$ItemName'. Ensure you are logged in (run 'lpass login')."
}

$secret = $secret.Trim()

if ($PrintOnly) {
    Write-Output $secret
    exit 0
}

Set-Item -Path "Env:$EnvVar" -Value $secret
Write-Output "Set $EnvVar for this session from LastPass item '$ItemName'."

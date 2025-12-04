# Creates a dated potential feature file from the template and opens it plus backlog.md.
param(
    [string] $ShortName
)

if ([string]::IsNullOrWhiteSpace($ShortName)) {
    Write-Host 'Aborted: no name provided. (Pass -ShortName or use the VS Code task prompt.)'
    exit 1
}

$shortPattern = '^[a-z0-9]+(-[a-z0-9]+)*$'
if ($ShortName -notmatch $shortPattern) {
    Write-Host "Aborted: '$ShortName' is invalid. Use kebab-case letters/numbers only (e.g., notes-feature)."
    exit 1
}

$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$today = Get-Date -Format 'yyyy-MM-dd'
$target = Join-Path $workspace "docs/features/potential/$today-$ShortName.md"
$template = Join-Path $workspace 'docs/features/potential/template.md'
$backlog = Join-Path $workspace 'docs/features/backlog.md'

Copy-Item $template $target -Force
Write-Host "Created: $target"

# Populate placeholders in the new file
$author = (git config user.name) 2>$null
if (-not $author -or [string]::IsNullOrWhiteSpace($author)) {
    $author = $env:USERNAME
}
if (-not $author) { $author = 'Unknown' }

$content = Get-Content -Raw -Path $target
$content = $content -replace '<feature-name>', $ShortName
$content = $content -replace 'YYYY-MM-DD', $today
$content = $content -replace '- Author: name', "- Author: $author"
Set-Content -Path $target -Value $content -Encoding UTF8

$codeCmd = Get-Command code -ErrorAction SilentlyContinue
if ($codeCmd) {
    Start-Process code -ArgumentList @($target, $backlog)
} else {
    Write-Warning "VS Code 'code' command not found. Open files manually:"
    Write-Host "  $target"
    Write-Host "  $backlog"
}

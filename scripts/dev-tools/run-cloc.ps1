param(
    [string]$Path = "$PSScriptRoot/.."
)

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSStyle.OutputRendering = 'PlainText'
}

$root = Resolve-Path $Path
$toolsDir = Join-Path $PSScriptRoot ".." "tools"
$clocExe = Join-Path $toolsDir "cloc.exe"
$clocScript = Join-Path $toolsDir "cloc"
$args = @("--vcs=git", "--quiet", "--exclude-dir=tools", $root)

$onWindows = if ($PSVersionTable.PSVersion.Major -ge 6) {
    $IsWindows
} else {
    $env:OS -eq "Windows_NT"
}

if ($onWindows -and (Test-Path $clocExe)) {
    & $clocExe @args
}
elseif (Test-Path $clocScript) {
    $perl = Get-Command perl -ErrorAction SilentlyContinue
    if (-not $perl) {
        throw "Perl is required to run the bundled cloc script."
    }
    & $perl.Path $clocScript @args
}
else {
    throw "Bundled cloc binary not found."
}

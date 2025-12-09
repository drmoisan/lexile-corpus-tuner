param(
    [string]$Path = "$PSScriptRoot/.."
)

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSStyle.OutputRendering = 'PlainText'
}

$root = Resolve-Path $Path
$toolsRoot = Join-Path -Path $PSScriptRoot -ChildPath ".."
$toolsDir = Join-Path -Path $toolsRoot -ChildPath "tools"
$clocExe = Join-Path -Path $toolsDir -ChildPath "cloc.exe"
$clocScript = Join-Path -Path $toolsDir -ChildPath "cloc"
$clocArgs = @("--vcs=git", "--quiet", "--exclude-dir=tools", $root)

$onWindows = if ($PSVersionTable.PSVersion.Major -ge 6) {
    $IsWindows
} else {
    $env:OS -eq "Windows_NT"
}

if ($onWindows -and (Test-Path $clocExe)) {
    & $clocExe @clocArgs
}
elseif (Test-Path $clocScript) {
    $perl = Get-Command perl -ErrorAction SilentlyContinue
    if (-not $perl) {
        throw "Perl is required to run the bundled cloc script."
    }
    & $perl.Path $clocScript @clocArgs
}
else {
    throw "Bundled cloc binary not found."
}

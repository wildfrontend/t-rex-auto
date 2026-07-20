[CmdletBinding()]
param(
    [ValidateSet("runtime", "debug", "training")]
    [string]$Mode = "runtime",
    [int]$MaxActions = 0
)

$ErrorActionPreference = "Stop"
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "DinoMutantBot"
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$AppRoot = Join-Path $RuntimeRoot "app"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run setup-windows.ps1 first."
}

& $PythonExecutable (Join-Path $AppRoot "main.py") `
    --config (Join-Path $AppRoot "config.json") `
    run --mode $Mode --max-actions $MaxActions
exit $LASTEXITCODE

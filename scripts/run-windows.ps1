[CmdletBinding()]
param(
    [ValidateSet("runtime", "debug", "training")]
    [string]$Mode = "runtime",
    [int]$MaxActions = 0,
    [int]$MaxCycles = 0,
    [int]$BatchSize = 0
)

$ErrorActionPreference = "Stop"
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "DinoMutantBot"
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$AppRoot = Join-Path $RuntimeRoot "app"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run setup-windows.ps1 first."
}

$RunArguments = @(
    (Join-Path $AppRoot "main.py"),
    "--config", (Join-Path $AppRoot "config.json"),
    "run", "--mode", $Mode,
    "--max-actions", $MaxActions,
    "--max-cycles", $MaxCycles
)
if ($BatchSize -gt 0) {
    $RunArguments += @("--batch-size", $BatchSize)
}

& $PythonExecutable $RunArguments
exit $LASTEXITCODE

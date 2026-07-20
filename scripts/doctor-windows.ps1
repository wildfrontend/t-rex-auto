[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RuntimeRoot = "D:\DinoMutantBot"
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$AppRoot = Join-Path $RuntimeRoot "app"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run setup-windows.ps1 first."
}
& $PythonExecutable (Join-Path $AppRoot "main.py") `
    --config (Join-Path $AppRoot "config.json") doctor
exit $LASTEXITCODE

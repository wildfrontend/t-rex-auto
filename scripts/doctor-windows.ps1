[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-bot.cmd for guided setup."
}
$DoctorExitCode = 1
Push-Location $RuntimeRoot
try {
    & $PythonExecutable (Join-Path $AppRoot "main.py") `
        --config (Join-Path $AppRoot "config.json") doctor
    $DoctorExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $DoctorExitCode

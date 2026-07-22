[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
if (-not (Test-Path $PythonExecutable)) {
    $SharedPython = "D:\DinoMutantBot\python\python.exe"
    if (Test-Path $SharedPython) {
        $PythonExecutable = $SharedPython
        Write-Host "Using shared Python runtime: $SharedPython"
    }
}
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run setup-windows.ps1 first."
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

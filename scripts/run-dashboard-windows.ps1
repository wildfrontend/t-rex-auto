[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8780
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$MainScript = Join-Path $AppRoot "main.py"
$ConfigPath = Join-Path $AppRoot "config.json"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-bot.cmd for guided setup."
}
if (-not (Test-Path -LiteralPath $MainScript)) {
    throw "Dashboard entrypoint not found: $MainScript"
}

Write-Host "Dino Mutant Bot - Web Dashboard" -ForegroundColor Cyan
Write-Host "Dashboard: http://127.0.0.1:$Port"
Write-Host "Closing this window only closes the dashboard; it does not stop the Bot."

& $PythonExecutable `
    $MainScript `
    "--config" $ConfigPath `
    "dashboard" `
    "--port" $Port `
    "--open-browser"
exit $LASTEXITCODE

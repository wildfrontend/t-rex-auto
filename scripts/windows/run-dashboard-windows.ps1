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
$DashboardUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-dashboard.cmd for guided setup."
}
if (-not (Test-Path -LiteralPath $MainScript)) {
    throw "Dashboard entrypoint not found: $MainScript"
}

function Remove-LegacyLoginStartup {
    $StartupDirectory = [Environment]::GetFolderPath("Startup")
    if ([string]::IsNullOrWhiteSpace($StartupDirectory)) {
        return
    }
    $StartupEntry = Join-Path $StartupDirectory "Dino Dashboard Server.cmd"
    if (Test-Path -LiteralPath $StartupEntry) {
        Remove-Item -LiteralPath $StartupEntry -Force
        Write-Host "Removed legacy Dashboard login startup entry." -ForegroundColor Yellow
    }
}

Remove-LegacyLoginStartup

$DashboardArguments = @(
    $MainScript,
    "--config", $ConfigPath,
    "dashboard",
    "--port", [string]$Port,
    "--open-browser"
)

Write-Host "Dashboard is running in this window: $DashboardUrl" -ForegroundColor Green
Write-Host "Close this window or press Ctrl+C to stop Dashboard." -ForegroundColor Cyan

$DashboardExitCode = 1
Push-Location $RuntimeRoot
try {
    & $PythonExecutable @DashboardArguments
    $DashboardExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    Write-Host "Dashboard stopped." -ForegroundColor Yellow
}
exit $DashboardExitCode

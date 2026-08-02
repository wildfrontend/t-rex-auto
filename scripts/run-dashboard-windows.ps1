[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8780,
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$MainScript = Join-Path $AppRoot "main.py"
$ConfigPath = Join-Path $AppRoot "config.json"
$WatcherScript = Join-Path $PSScriptRoot "watch-dashboard-windows.ps1"
$DashboardUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-bot.cmd for guided setup."
}
if (-not (Test-Path -LiteralPath $MainScript)) {
    throw "Dashboard entrypoint not found: $MainScript"
}
if (-not (Test-Path -LiteralPath $WatcherScript)) {
    throw "Dashboard watcher not found: $WatcherScript"
}

function Test-DashboardHealth {
    try {
        $Health = Invoke-RestMethod `
            -Uri "$DashboardUrl/api/health" `
            -TimeoutSec 2
        return $Health.ok -eq $true -and $Health.service -eq "dino-dashboard"
    } catch {
        return $false
    }
}

function Install-LoginStartup {
    $StartupDirectory = [Environment]::GetFolderPath("Startup")
    if ([string]::IsNullOrWhiteSpace($StartupDirectory)) {
        Write-Warning "Unable to locate the current user's Startup folder."
        return
    }
    $Launcher = Join-Path $RuntimeRoot "start-dashboard.cmd"
    $StartupEntry = Join-Path $StartupDirectory "Dino Dashboard Server.cmd"
    $Contents = "@echo off`r`ncall `"$Launcher`" --server-only`r`n"
    $Encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($StartupEntry, $Contents, $Encoding)
}

$WatcherArguments = @(
    "-NoLogo",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$WatcherScript`"",
    "-Port", [string]$Port
)
Start-Process powershell.exe `
    -ArgumentList $WatcherArguments `
    -WorkingDirectory $RuntimeRoot `
    -WindowStyle Hidden

if (-not $ServerOnly) {
    Install-LoginStartup
}

$Ready = $false
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    if (Test-DashboardHealth) {
        $Ready = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $Ready) {
    throw "Dashboard did not become ready at $DashboardUrl"
}

if (-not $ServerOnly) {
    Start-Process $DashboardUrl
}
Write-Host "Dashboard is running in the background: $DashboardUrl" -ForegroundColor Green
exit 0

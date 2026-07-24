[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("status", "start", "stop", "restart", "doctor", "diagnostics", "snapshot")]
    [string]$Action,
    [ValidateSet("fast", "safe")]
    [string]$Speed = "fast",
    [ValidateRange(1, 65535)]
    [int]$StatusPort = 8765,
    [switch]$Confirm
)

$ErrorActionPreference = "Stop"
$Utf8Encoding = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8Encoding
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$LauncherScript = Join-Path $PSScriptRoot "launcher-windows.ps1"
$ApiRoot = "http://127.0.0.1:$StatusPort"

function Write-JsonResult {
    param([hashtable]$Value)
    $Value | ConvertTo-Json -Depth 8 -Compress
}

function Assert-MutationConfirmed {
    if (-not $Confirm) {
        Write-JsonResult @{
            ok = $false
            error = "confirmation_required"
            message = "start, stop, and restart require -Confirm"
        }
        exit 2
    }
}

function Get-BotProcesses {
    $MainScript = [regex]::Escape((Join-Path $AppRoot "main.py"))
    return @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object {
                $_.CommandLine -match $MainScript -and
                $_.CommandLine -match " run "
            }
    )
}

function Resolve-PythonExecutable {
    $LocalPython = Join-Path $RuntimeRoot "python\python.exe"
    if (Test-Path $LocalPython) {
        return $LocalPython
    }
    throw "Python runtime not found at $LocalPython"
}

function Get-ApiStatus {
    return Invoke-RestMethod -Uri "$ApiRoot/status" -TimeoutSec 3
}

function Get-ApiHealth {
    return Invoke-RestMethod -Uri "$ApiRoot/health" -TimeoutSec 3
}

function Assert-DinoBotApiIdentity {
    param([switch]$RequireProcessIdentity)
    $Health = Get-ApiHealth
    if ($Health.ok -ne $true -or $Health.service -ne "dino-mutant-bot-status") {
        throw "status_api_identity_mismatch"
    }
    if (-not $RequireProcessIdentity) {
        return $Health
    }
    if ($null -eq $Health.process_id) {
        throw "status_api_process_identity_missing"
    }
    $ApiProcessId = [int]$Health.process_id
    $Connection = Get-NetTCPConnection `
        -LocalPort $StatusPort `
        -State Listen `
        -ErrorAction Stop |
        Where-Object { $_.OwningProcess -eq $ApiProcessId } |
        Select-Object -First 1
    if ($null -eq $Connection) {
        throw "status_api_process_identity_mismatch"
    }
    $Process = Get-CimInstance `
        Win32_Process `
        -Filter "ProcessId=$ApiProcessId" `
        -ErrorAction Stop
    $PortPattern = "--status-port\s+" + [regex]::Escape([string]$StatusPort) + "(?:\s|$)"
    if (
        $Process.Name -ine "python.exe" -or
        $Process.CommandLine -notmatch "main\.py" -or
        $Process.CommandLine -notmatch " run " -or
        $Process.CommandLine -notmatch $PortPattern
    ) {
        throw "status_api_process_identity_mismatch"
    }
    return $Health
}

function Request-GracefulStop {
    [void](Assert-DinoBotApiIdentity -RequireProcessIdentity)
    return Invoke-RestMethod `
        -Method Post `
        -Uri "$ApiRoot/control/stop" `
        -TimeoutSec 3
}

function Start-BotLauncher {
    if ((Get-BotProcesses).Count -gt 0) {
        return @{ ok = $true; action = "start"; result = "already_running" }
    }
    $Arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$LauncherScript`"",
        "-Speed", $Speed,
        "-StatusPort", [string]$StatusPort
    )
    $Process = Start-Process powershell.exe `
        -ArgumentList $Arguments `
        -WorkingDirectory $RuntimeRoot `
        -PassThru
    return @{
        ok = $true
        action = "start"
        result = "launcher_started"
        launcher_pid = $Process.Id
        speed = $Speed
        status_port = $StatusPort
    }
}

try {
    if ($Action -eq "status") {
        try {
            [void](Assert-DinoBotApiIdentity)
            Get-ApiStatus | ConvertTo-Json -Depth 8
        } catch {
            $ErrorCode = if ($_.Exception.Message -like "status_api_*") {
                $_.Exception.Message
            } else {
                "status_api_unavailable"
            }
            Write-JsonResult @{
                ok = $false
                running = $false
                error = $ErrorCode
                message = $_.Exception.Message
                url = "$ApiRoot/status"
            }
            exit 1
        }
    } elseif ($Action -eq "start") {
        Assert-MutationConfirmed
        Write-JsonResult (Start-BotLauncher)
    } elseif ($Action -eq "stop") {
        Assert-MutationConfirmed
        $Response = Request-GracefulStop
        Write-JsonResult @{
            ok = $true
            action = "stop"
            result = "graceful_stop_requested"
            response = $Response
        }
    } elseif ($Action -eq "restart") {
        Assert-MutationConfirmed
        [void](Request-GracefulStop)
        $Deadline = (Get-Date).AddSeconds(20)
        while ((Get-BotProcesses).Count -gt 0 -and (Get-Date) -lt $Deadline) {
            Start-Sleep -Milliseconds 500
        }
        if ((Get-BotProcesses).Count -gt 0) {
            throw "Bot did not stop within 20 seconds"
        }
        $Result = Start-BotLauncher
        $Result.action = "restart"
        Write-JsonResult $Result
    } elseif ($Action -eq "doctor") {
        $PythonExecutable = Resolve-PythonExecutable
        & $PythonExecutable `
            (Join-Path $AppRoot "main.py") `
            "--config" (Join-Path $AppRoot "config.json") `
            "doctor"
        exit $LASTEXITCODE
    } elseif ($Action -eq "diagnostics") {
        $PythonExecutable = Resolve-PythonExecutable
        $BundlePath = Join-Path $AppRoot (
            "diagnostics\dino-diagnostic-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".zip"
        )
        & $PythonExecutable `
            (Join-Path $AppRoot "main.py") `
            "--config" (Join-Path $AppRoot "config.json") `
            "diagnostics" "--output" $BundlePath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Diagnostic bundle creation failed"
        }
        Write-JsonResult @{
            ok = $true
            action = "diagnostics"
            output = $BundlePath
            includes_screenshot = $false
        }
    } elseif ($Action -eq "snapshot") {
        $PythonExecutable = Resolve-PythonExecutable
        $Output = Join-Path $AppRoot ("debug\ai-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".png")
        & $PythonExecutable `
            (Join-Path $AppRoot "main.py") `
            "--config" (Join-Path $AppRoot "config.json") `
            "snapshot" "--backend" "adb" "--output" $Output
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        Write-JsonResult @{ ok = $true; action = "snapshot"; output = $Output }
    }
} catch {
    Write-JsonResult @{
        ok = $false
        action = $Action
        error = $_.Exception.Message
    }
    exit 1
}

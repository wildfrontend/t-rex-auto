[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("status", "start", "stop", "restart", "restart-game", "doctor", "diagnostics", "snapshot")]
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
$MainScript = Join-Path $AppRoot "main.py"
$ConfigPath = Join-Path $AppRoot "config.json"
$StopWaitSeconds = 20
$StartWaitSeconds = 60

function Write-JsonResult {
    param([hashtable]$Value)
    $Value | ConvertTo-Json -Depth 8 -Compress
}

function Throw-ControlError {
    param(
        [string]$Code,
        [string]$Message
    )
    $Exception = New-Object System.InvalidOperationException $Message
    $Exception.Data["DinoErrorCode"] = $Code
    throw $Exception
}

function Assert-MutationConfirmed {
    if (-not $Confirm) {
        Write-JsonResult @{
            ok = $false
            error = "confirmation_required"
            message = "start, stop, restart, and restart-game require -Confirm"
        }
        exit 2
    }
}

function Get-BotProcesses {
    $MainScriptPattern = [regex]::Escape($MainScript)
    $ConfigPattern = [regex]::Escape($ConfigPath)
    return @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object {
                $_.CommandLine -match $MainScriptPattern -and
                $_.CommandLine -match " run " -and
                $_.CommandLine -match $ConfigPattern
            }
    )
}

function Test-ProcessExists {
    param([int]$ProcessId)
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Test-StatusPortAvailable {
    param([int]$Port)
    $Listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    try {
        $Listener.Start()
        return $true
    } catch [Net.Sockets.SocketException] {
        return $false
    } finally {
        $Listener.Stop()
    }
}

function Get-StatusPortOwner {
    param([int]$Port)
    $Connection = Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $Connection) {
        return $null
    }
    $OwnerProcessId = [int]$Connection.OwningProcess
    $Process = Get-CimInstance `
        Win32_Process `
        -Filter "ProcessId=$OwnerProcessId" `
        -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        ProcessId = $OwnerProcessId
        Name = if ($null -ne $Process) { [string]$Process.Name } else { "unknown" }
        CommandLine = if ($null -ne $Process) { [string]$Process.CommandLine } else { "unavailable" }
    }
}

function Format-StatusPortOwner {
    param([int]$Port)
    $Owner = Get-StatusPortOwner $Port
    if ($null -eq $Owner) {
        return "Port $Port 沒有可辨識的 LISTEN PID，但目前仍無法綁定"
    }
    return "PID $($Owner.ProcessId) / $($Owner.Name) / $($Owner.CommandLine)"
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
    try {
        $Health = Get-ApiHealth
    } catch {
        Throw-ControlError `
            "status_api_unavailable" `
            "Port $StatusPort 無法讀取 Dino Bot /health：$($_.Exception.Message)"
    }
    if ($Health.ok -ne $true -or $Health.service -ne "dino-mutant-bot-status") {
        Throw-ControlError `
            "status_api_identity_mismatch" `
            "Port $StatusPort 的 API 身分不符：service=$($Health.service)，ok=$($Health.ok)"
    }
    if (-not $RequireProcessIdentity) {
        return $Health
    }
    if ($null -eq $Health.process_id) {
        Throw-ControlError `
            "status_api_process_identity_missing" `
            "Port $StatusPort 的 Dino Bot API 未回報 process_id"
    }
    $ApiProcessId = [int]$Health.process_id
    $Connection = Get-NetTCPConnection `
        -LocalPort $StatusPort `
        -State Listen `
        -ErrorAction Stop |
        Where-Object { $_.OwningProcess -eq $ApiProcessId } |
        Select-Object -First 1
    if ($null -eq $Connection) {
        Throw-ControlError `
            "status_api_process_identity_mismatch" `
            "API 回報 PID $ApiProcessId，但該 PID 並未監聽 Port $StatusPort"
    }
    $Process = Get-CimInstance `
        Win32_Process `
        -Filter "ProcessId=$ApiProcessId" `
        -ErrorAction Stop
    $PortPattern = "--status-port\s+" + [regex]::Escape([string]$StatusPort) + "(?:\s|$)"
    $MainPattern = [regex]::Escape($MainScript)
    $ConfigPattern = [regex]::Escape($ConfigPath)
    if (
        $Process.Name -ine "python.exe" -or
        $Process.CommandLine -notmatch $MainPattern -or
        $Process.CommandLine -notmatch $ConfigPattern -or
        $Process.CommandLine -notmatch " run " -or
        $Process.CommandLine -notmatch $PortPattern
    ) {
        Throw-ControlError `
            "status_api_process_identity_mismatch" `
            "Port $StatusPort 的 PID $ApiProcessId 命令列不符合此版本 Bot：$($Process.CommandLine)"
    }
    return $Health
}

function Request-GracefulStop {
    [void](Assert-DinoBotApiIdentity -RequireProcessIdentity)
    $Response = Invoke-RestMethod `
        -Method Post `
        -Uri "$ApiRoot/control/stop" `
        -TimeoutSec 3
    if ($Response.accepted -ne $true -or $Response.action -ne "stop") {
        Throw-ControlError `
            "stop_not_accepted" `
            "已驗證的 Dino Bot 拒絕停止要求：$($Response | ConvertTo-Json -Compress)"
    }
    return $Response
}

function Request-GameRestart {
    [void](Assert-DinoBotApiIdentity -RequireProcessIdentity)
    return Invoke-RestMethod `
        -Method Post `
        -Uri "$ApiRoot/control/restart-game" `
        -TimeoutSec 3
}

function Assert-StatusPortStartable {
    if (Test-StatusPortAvailable $StatusPort) {
        return $null
    }
    try {
        return Assert-DinoBotApiIdentity -RequireProcessIdentity
    } catch {
        $Owner = Format-StatusPortOwner $StatusPort
        $Reason = $_.Exception.Message
        Throw-ControlError `
            "port_occupied_unverified" `
            "Port $StatusPort 已被占用（$Owner），且無法驗證為本版本 Dino Bot：$Reason"
    }
}

function Wait-ForCleanStop {
    param(
        [int]$ExpectedProcessId,
        [int]$TimeoutSeconds = $StopWaitSeconds
    )
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        $Owner = Get-StatusPortOwner $StatusPort
        if ($null -ne $Owner -and [int]$Owner.ProcessId -ne $ExpectedProcessId) {
            Throw-ControlError `
                "port_reoccupied" `
                "舊 Bot 停止期間 Port $StatusPort 被其他程序占用：$(Format-StatusPortOwner $StatusPort)；未啟動新 Bot"
        }
        if (
            -not (Test-ProcessExists $ExpectedProcessId) -and
            (Test-StatusPortAvailable $StatusPort)
        ) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    if (Test-ProcessExists $ExpectedProcessId) {
        $PortState = if (Test-StatusPortAvailable $StatusPort) {
            "Port 已釋放，但程序仍存在"
        } else {
            Format-StatusPortOwner $StatusPort
        }
        Throw-ControlError `
            "stop_timeout" `
            "已送出停止要求，但舊 Bot PID $ExpectedProcessId 在 $TimeoutSeconds 秒後仍未退出（$PortState）；未啟動新 Bot"
    }
    Throw-ControlError `
        "port_not_reusable" `
        "舊 Bot PID $ExpectedProcessId 已退出，但 Port $StatusPort 在 $TimeoutSeconds 秒後仍無法重新綁定：$(Format-StatusPortOwner $StatusPort)；未啟動新 Bot"
}

function Wait-ForBotReady {
    param(
        [System.Diagnostics.Process]$LauncherProcess,
        [int]$TimeoutSeconds = $StartWaitSeconds
    )
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $LastReason = "新程序尚未監聽 status port"
    while ((Get-Date) -lt $Deadline) {
        $LauncherProcess.Refresh()
        if ($LauncherProcess.HasExited) {
            Throw-ControlError `
                "launcher_exited" `
                "Windows launcher PID $($LauncherProcess.Id) 在 Bot API 就緒前退出（exit code $($LauncherProcess.ExitCode)）"
        }
        if (-not (Test-StatusPortAvailable $StatusPort)) {
            try {
                $Health = Assert-DinoBotApiIdentity -RequireProcessIdentity
                $Status = Get-ApiStatus
                if ($Status.running -ne $false) {
                    return @{
                        health = $Health
                        status = $Status
                    }
                }
                $LastReason = "status API 已回應，但 running=false"
            } catch {
                $LastReason = $_.Exception.Message
            }
        }
        Start-Sleep -Milliseconds 250
    }
    $PortState = if (Test-StatusPortAvailable $StatusPort) {
        "沒有程序監聽"
    } else {
        Format-StatusPortOwner $StatusPort
    }
    Throw-ControlError `
        "status_api_start_timeout" `
        "Windows launcher PID $($LauncherProcess.Id) 已啟動，但 $TimeoutSeconds 秒內 Bot API 未就緒；Port $StatusPort：$PortState；最後原因：$LastReason"
}

function Start-BotLauncher {
    $ExistingHealth = Assert-StatusPortStartable
    if ($null -ne $ExistingHealth) {
        return @{
            ok = $true
            action = "start"
            result = "already_running"
            process_id = [int]$ExistingHealth.process_id
            feature = $ExistingHealth.feature
            status_port = $StatusPort
        }
    }
    $ExistingBots = @(Get-BotProcesses)
    if ($ExistingBots.Count -gt 0) {
        $Details = ($ExistingBots | ForEach-Object {
            "PID $($_.ProcessId) / $($_.CommandLine)"
        }) -join "；"
        Throw-ControlError `
            "config_already_running" `
            "同一個 config.json 已由其他 Bot 使用，但它不在預期 Port $StatusPort：$Details"
    }
    $Arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$LauncherScript`"",
        "-Speed", $Speed,
        "-StatusPort", [string]$StatusPort,
        "-SkipEmulatorPrompt"
    )
    $Process = Start-Process powershell.exe `
        -ArgumentList $Arguments `
        -WorkingDirectory $RuntimeRoot `
        -PassThru
    $Ready = Wait-ForBotReady -LauncherProcess $Process
    return @{
        ok = $true
        action = "start"
        result = "started"
        launcher_pid = $Process.Id
        process_id = [int]$Ready.health.process_id
        feature = $Ready.health.feature
        speed = $Speed
        status_port = $StatusPort
        health = $Ready.health
        status = $Ready.status
    }
}

try {
    if ($Action -eq "status") {
        try {
            [void](Assert-DinoBotApiIdentity)
            Get-ApiStatus | ConvertTo-Json -Depth 8
        } catch {
            $ErrorCode = if ($_.Exception.Data.Contains("DinoErrorCode")) {
                [string]$_.Exception.Data["DinoErrorCode"]
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
        $Health = Assert-DinoBotApiIdentity -RequireProcessIdentity
        $ProcessId = [int]$Health.process_id
        $Response = Request-GracefulStop
        Wait-ForCleanStop -ExpectedProcessId $ProcessId
        Write-JsonResult @{
            ok = $true
            action = "stop"
            result = "stopped"
            process_id = $ProcessId
            status_port = $StatusPort
            response = $Response
        }
    } elseif ($Action -eq "restart") {
        Assert-MutationConfirmed
        $Health = Assert-DinoBotApiIdentity -RequireProcessIdentity
        $ProcessId = [int]$Health.process_id
        [void](Request-GracefulStop)
        Wait-ForCleanStop -ExpectedProcessId $ProcessId
        $Result = Start-BotLauncher
        $Result.action = "restart"
        $Result.previous_process_id = $ProcessId
        Write-JsonResult $Result
    } elseif ($Action -eq "restart-game") {
        Assert-MutationConfirmed
        $Before = Get-ApiStatus
        $PreviousRestarts = [int]$Before.game_restarts
        $PreviousFailures = [int]$Before.game_restart_failures
        $Response = Request-GameRestart
        $Deadline = (Get-Date).AddSeconds(20)
        do {
            Start-Sleep -Milliseconds 500
            $Status = Get-ApiStatus
            if ([int]$Status.game_restart_failures -gt $PreviousFailures) {
                throw "game_restart_failed"
            }
            if ([int]$Status.game_restarts -gt $PreviousRestarts) {
                Write-JsonResult @{
                    ok = $true
                    action = "restart-game"
                    result = "game_restarted"
                    response = $Response
                    status = $Status
                }
                exit 0
            }
        } while ((Get-Date) -lt $Deadline)
        Write-JsonResult @{
            ok = $true
            action = "restart-game"
            result = "restart_requested"
            confirmation = "pending"
            response = $Response
            status = $Status
        }
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
            "diagnostics" "--include-screenshot" "--output" $BundlePath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Diagnostic bundle creation failed"
        }
        Write-JsonResult @{
            ok = $true
            action = "diagnostics"
            output = $BundlePath
            includes_screenshot = $true
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
    $ErrorCode = if ($_.Exception.Data.Contains("DinoErrorCode")) {
        [string]$_.Exception.Data["DinoErrorCode"]
    } else {
        "control_failed"
    }
    Write-JsonResult @{
        ok = $false
        action = $Action
        error = $ErrorCode
        message = $_.Exception.Message
        status_port = $StatusPort
    }
    exit 1
}

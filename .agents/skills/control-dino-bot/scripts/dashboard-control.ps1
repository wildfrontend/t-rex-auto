[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "status",
        "scan-adb",
        "start-hunt",
        "start-hatch-hunt",
        "start-stage-hatch",
        "start-stage-attack",
        "start-stage-hp",
        "start-stage-collect",
        "start-stage-cave",
        "stop",
        "restart-bot",
        "restart-game",
        "snapshot",
        "diagnostics"
    )]
    [string]$Action,
    [string]$Instance = "all",
    [switch]$Confirm
)

$ErrorActionPreference = "Stop"
$Utf8Encoding = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8Encoding
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$DashboardRoot = "http://127.0.0.1:8780"

function Write-JsonResult {
    param([object]$Value)
    $Value | ConvertTo-Json -Depth 12 -Compress
}

function Get-DashboardHealth {
    $Health = Invoke-RestMethod -Uri "$DashboardRoot/api/health" -TimeoutSec 3
    if ($Health.ok -ne $true -or $Health.service -ne "dino-dashboard") {
        throw "Dashboard identity mismatch: service=$($Health.service), ok=$($Health.ok)"
    }
    return $Health
}

function Get-DashboardOverview {
    param([string]$InstanceId)
    $Suffix = if ($InstanceId -and $InstanceId -ne "all") {
        "?instance=$([uri]::EscapeDataString($InstanceId))"
    } else {
        ""
    }
    return Invoke-RestMethod -Uri "$DashboardRoot/api/overview$Suffix" -TimeoutSec 8
}

function Convert-ToCompactInstance {
    param([object]$Item)
    $Active = $Item.active
    $Status = $Active.status
    $Workflow = $Active.workflow
    $Recent = @($Status.recent_actions)
    $LastAction = if ($Recent.Count -gt 0) { $Recent[-1] } else { $null }
    return [pscustomobject][ordered]@{
        id = $Item.id
        name = $Item.name
        serial = $Item.serial
        status_port = $Item.status_port
        allowed_modes = @($Item.allowed_modes)
        running = [bool]$Active.running
        feature = $Active.feature
        process_id = $Active.process_id
        current_stage = $Status.current_stage
        workflow_stage = $Workflow.stage
        workflow_label = $Workflow.label
        session_started = $Status.session_started
        last_log_time = $Status.last_log_time
        successful_hunts = $Status.successful_hunts
        total_actions = $Status.total_actions
        verification_failures = $Status.verification_failures
        black_screen_detections = $Status.black_screen_detections
        game_restarts = $Status.game_restarts
        last_successful_hunt = $Status.last_successful_hunt
        last_action = $LastAction
        operation = $Item.operation
    }
}

try {
    $Health = Get-DashboardHealth
    if ($Action -eq "status") {
        $Overview = Get-DashboardOverview -InstanceId $Instance
        $Items = @($Overview.instances)
        if ($Instance -ne "all") {
            $Items = @($Items | Where-Object { $_.id -eq $Instance })
        }
        if ($Items.Count -eq 0) {
            throw "Unknown Bot instance: $Instance"
        }
        $Compact = @($Items | ForEach-Object { Convert-ToCompactInstance $_ })
        $Collisions = @(
            $Compact |
                Where-Object { $_.running -and $_.serial } |
                Group-Object serial |
                Where-Object { $_.Count -gt 1 } |
                ForEach-Object { $_.Name }
        )
        Write-JsonResult ([ordered]@{
            ok = $true
            evidence = "dashboard_loopback"
            dashboard = [ordered]@{
                service = $Health.service
                api_version = $Health.api_version
                process_id = $Health.process_id
            }
            process_identity_verified = $false
            instances = $Compact
            serial_collisions = $Collisions
        })
        exit 0
    }

    if ($Instance -eq "all") {
        throw "A concrete -Instance is required for action '$Action'"
    }
    if ($Action -ne "scan-adb" -and -not $Confirm) {
        Write-JsonResult ([ordered]@{
            ok = $false
            error = "confirmation_required"
            action = $Action
            instance = $Instance
            message = "This action requires explicit user authorization and -Confirm"
        })
        exit 2
    }

    $EncodedInstance = [uri]::EscapeDataString($Instance)
    $Headers = @{
        "Origin" = $DashboardRoot
        "X-Dino-Dashboard" = "1"
    }
    $Response = Invoke-RestMethod `
        -Method Post `
        -Uri "$DashboardRoot/api/control/$Action`?instance=$EncodedInstance" `
        -Headers $Headers `
        -ContentType "application/json" `
        -Body "{}" `
        -TimeoutSec 15
    Write-JsonResult ([ordered]@{
        ok = $true
        evidence = "dashboard_loopback"
        dashboard_process_id = $Health.process_id
        instance = $Instance
        action = $Action
        control_identity_guarded_by_dashboard = $true
        response = $Response
    })
} catch {
    $Detail = $_.ErrorDetails.Message
    if (-not $Detail) {
        $Detail = $_.Exception.Message
    }
    Write-JsonResult ([ordered]@{
        ok = $false
        evidence = "dashboard_loopback"
        action = $Action
        instance = $Instance
        error = "dashboard_control_failed"
        message = $Detail
    })
    exit 1
}

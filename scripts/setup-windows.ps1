[CmdletBinding()]
param([string]$RuntimeRoot = "")

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $AppRoot = Split-Path -Parent $PSScriptRoot
    $RuntimeRoot = Split-Path -Parent $AppRoot
}
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"

function Invoke-Python {
    param(
        [string]$Description,
        [string[]]$PythonArguments
    )
    $Process = Start-Process -FilePath $PythonExecutable `
        -ArgumentList $PythonArguments -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        throw "$Description failed with exit code $($Process.ExitCode)"
    }
}

if (-not (Test-Path $PythonExecutable)) {
    throw "Portable runtime is not installed. Run install-windows-runtime.ps1 first."
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
Invoke-Python "pip upgrade" @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Python "dependency install" @(
    "-m", "pip", "install",
    "numpy>=2,<3", "opencv-python-headless>=4.10,<5", "mss>=9,<11",
    "pywin32>=306"
)

Write-Host "Windows environment ready: $PythonExecutable"
Write-Host "Enable Android Debug Bridge in BlueStacks, then run doctor-windows.ps1."

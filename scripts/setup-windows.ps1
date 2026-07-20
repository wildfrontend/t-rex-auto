[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "DinoMutantBot"
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"

if (-not (Test-Path $PythonExecutable)) {
    throw "Portable runtime is not installed. Run install-windows-runtime.ps1 first."
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
& $PythonExecutable -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }
& $PythonExecutable -m pip install `
    "numpy>=2,<3" "opencv-python-headless>=4.10,<5" "mss>=9,<11" `
    "pywin32>=306" "pytest>=8,<9" "ruff>=0.9,<1"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed with exit code $LASTEXITCODE" }

Write-Host "Windows environment ready: $PythonExecutable"
Write-Host "Enable Android Debug Bridge in BlueStacks, then run doctor-windows.ps1."

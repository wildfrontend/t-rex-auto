[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RuntimeRoot = "D:\DinoMutantBot"
$PythonRoot = Join-Path $RuntimeRoot "python"
$PythonExecutable = Join-Path $PythonRoot "python.exe"
$Archive = Join-Path $env:TEMP "python-3.12.10-embed-amd64.zip"
$Bootstrap = Join-Path $env:TEMP "get-pip.py"
$ExpectedMd5 = "FE8EF205F2E9C3BA44D0CF9954E1ABD3"

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

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
Invoke-WebRequest -UseBasicParsing `
    -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" `
    -OutFile $Archive
if ((Get-FileHash $Archive -Algorithm MD5).Hash -ne $ExpectedMd5) {
    throw "Python archive checksum verification failed."
}
if (Test-Path $PythonRoot) {
    Remove-Item -Recurse -Force $PythonRoot
}
Expand-Archive -Path $Archive -DestinationPath $PythonRoot
Copy-Item (Join-Path $PSScriptRoot "python312._pth") (Join-Path $PythonRoot "python312._pth")

Invoke-WebRequest -UseBasicParsing -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $Bootstrap
Invoke-Python "pip bootstrap" @($Bootstrap, "--disable-pip-version-check")
Invoke-Python "pip upgrade" @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Python "dependency install" @(
    "-m", "pip", "install",
    "numpy>=2,<3", "opencv-python-headless>=4.10,<5", "mss>=9,<11",
    "pywin32>=306", "pytest>=8,<9", "ruff>=0.9,<1"
)

Write-Host "Portable Windows runtime ready: $PythonExecutable"
Write-Host "Run scripts/deploy-windows.sh from WSL before the first doctor check."

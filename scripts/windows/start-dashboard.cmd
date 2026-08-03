@echo off
setlocal
title 猛龍計畫 - Dashboard

if /I "%~1"=="uninstall" (
  set "uninstaller=%~dp0app\scripts\uninstall-windows.ps1"
  if not exist "%~dp0app\scripts\uninstall-windows.ps1" (
    echo ERROR: Uninstaller not found: %~dp0app\scripts\uninstall-windows.ps1
    pause
    exit /b 1
  )
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0app\scripts\uninstall-windows.ps1" ^
    -RuntimeRoot "%~dp0."
  if errorlevel 1 (
    echo.
    echo Uninstall did not complete.
    pause
    exit /b 1
  )
  exit /b 0
)

set "dashboard_runner=%~dp0app\scripts\run-dashboard-windows.ps1"
if not exist "%dashboard_runner%" (
  echo ERROR: Dashboard runner not found: %dashboard_runner%
  pause
  exit /b 1
)

set "runtime_python=%~dp0python\python.exe"
set "runtime_installer=%~dp0app\scripts\install-windows-runtime.ps1"
if not exist "%runtime_python%" (
  if not exist "%runtime_installer%" (
    echo ERROR: Windows runtime installer not found: %runtime_installer%
    pause
    exit /b 1
  )
  echo Windows runtime is not installed. Starting guided setup...
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%runtime_installer%" ^
    -RuntimeRoot "%~dp0."
  if errorlevel 1 (
    echo.
    echo Windows runtime installation failed.
    pause
    exit /b 1
  )
)

set "dashboard_port=%~1"
if "%dashboard_port%"=="" set "dashboard_port=8780"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%dashboard_runner%" ^
  -Port "%dashboard_port%"
set "dashboard_exit_code=%ERRORLEVEL%"

if not "%dashboard_exit_code%"=="0" (
  echo.
  echo Dashboard launcher failed with exit code %dashboard_exit_code%.
  pause
)
exit /b %dashboard_exit_code%

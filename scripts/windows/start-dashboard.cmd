@echo off
setlocal
title Dino Mutant Bot - Dashboard

set "dashboard_runner=%~dp0app\scripts\run-dashboard-windows.ps1"
if not exist "%dashboard_runner%" (
  echo ERROR: Dashboard runner not found: %dashboard_runner%
  pause
  exit /b 1
)

set "dashboard_port=%~1"
set "dashboard_mode="
if /I "%~1"=="--server-only" (
  set "dashboard_port=8780"
  set "dashboard_mode=-ServerOnly"
) else (
  if "%dashboard_port%"=="" set "dashboard_port=8780"
)

if defined dashboard_mode (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%dashboard_runner%" ^
    -Port "%dashboard_port%" ^
    %dashboard_mode%
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%dashboard_runner%" ^
    -Port "%dashboard_port%"
)
set "dashboard_exit_code=%ERRORLEVEL%"

if not "%dashboard_exit_code%"=="0" (
  echo.
  echo Dashboard launcher failed with exit code %dashboard_exit_code%.
  pause
)
exit /b %dashboard_exit_code%

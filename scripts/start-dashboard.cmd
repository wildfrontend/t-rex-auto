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
if "%dashboard_port%"=="" set "dashboard_port=8780"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%dashboard_runner%" ^
  -Port "%dashboard_port%"
set "dashboard_exit_code=%ERRORLEVEL%"

echo.
echo Dashboard closed with exit code %dashboard_exit_code%.
pause
exit /b %dashboard_exit_code%

@echo off
setlocal
title Dino Mutant Bot - Control

set "bot_launcher=%~dp0app\scripts\launcher-windows.ps1"
if not exist "%bot_launcher%" (
  echo ERROR: Interactive launcher not found: %bot_launcher%
  pause
  exit /b 1
)

if "%~1"=="" (
  if "%~2"=="" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%bot_launcher%"
  ) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%bot_launcher%" -StatusPort "%~2"
  )
) else (
  if "%~2"=="" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%bot_launcher%" -Speed "%~1"
  ) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%bot_launcher%" -Speed "%~1" -StatusPort "%~2"
  )
)
set "launcher_exit_code=%ERRORLEVEL%"

echo.
echo Launcher closed with exit code %launcher_exit_code%.
pause
exit /b %launcher_exit_code%

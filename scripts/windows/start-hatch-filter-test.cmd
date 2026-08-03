@echo off
setlocal
title 猛龍計畫 - Hatch Filter Test

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo.
echo SAFE TEST: Set nest tag filter to Attack specialization.
echo Open My Nest in the game before continuing.
echo This test will not replace dinosaurs, auto-place, or collect eggs.
echo.
pause

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-filter-test" ^
  -Mode "debug" ^
  -Speed "safe" ^
  -StatusPort "8767" ^
  -MaxActions "0" ^
  -MaxCycles "1"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Hatch filter test closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%

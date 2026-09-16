@echo off
setlocal
title 猛龍計畫 - Hatch Parent Open Test

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo.
echo SAFE TEST: Read both attack parents and open only the left parent.
echo Open My Nest with the Attack specialization filter collapsed before continuing.
echo The test stops on Select Dino and will not select or replace any dinosaur.
echo.
pause

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-parent-test" ^
  -Mode "debug" ^
  -Speed "safe" ^
  -StatusPort "8769" ^
  -MaxActions "1" ^
  -MaxCycles "1"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Hatch parent test closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%

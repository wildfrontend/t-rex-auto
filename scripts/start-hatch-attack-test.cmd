@echo off
setlocal
title Dino Mutant Bot - Hatch Attack Replacement Test

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo.
echo ATTACK TEST: Process both parents with all-tags and attack-descending sorting.
echo Open My Nest with its tag filter collapsed before continuing.
echo The test first sets the nest filter to Attack specialization.
echo A candidate is selected and confirmed ONLY when its attack is strictly higher.
echo This test can replace a parent when a valid upgrade exists.
echo.
pause

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-attack-test" ^
  -Mode "debug" ^
  -Speed "safe" ^
  -StatusPort "8770" ^
  -MaxActions "20" ^
  -MaxCycles "2"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Hatch attack test closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%

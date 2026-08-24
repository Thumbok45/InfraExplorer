@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo.
    echo InfraExplorer needs Python 3 on this PC.
    echo Install it from https://www.python.org/downloads/
    echo During setup, check "Add python.exe to PATH".
    echo Then double-click Launch.bat again.
    echo.
    pause
    exit /b 1
  )
)

%PY% InfraExplorer.py
if errorlevel 1 (
  echo.
  echo InfraExplorer exited with an error.
  pause
)

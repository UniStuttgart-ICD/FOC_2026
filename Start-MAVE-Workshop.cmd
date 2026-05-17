@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found on PATH.
  echo Install uv, then run this launcher again.
  pause
  exit /b 1
)

if not exist "server\.venv\Scripts\python.exe" (
  echo Creating the server environment...
  pushd server
  uv sync
  if errorlevel 1 (
    popd
    echo uv sync failed.
    pause
    exit /b 1
  )
  popd
)

pushd server
uv run python ..\scripts\run_operator_dashboard.py
set EXIT_CODE=%ERRORLEVEL%
popd

if not "%EXIT_CODE%"=="0" (
  echo Operator dashboard exited with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%

@echo off
setlocal

cd /d "%~dp0"

set "INSTALL_UR_RTDE=0"
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--with-ur-rtde" (
  set "INSTALL_UR_RTDE=1"
  shift
  goto parse_args
)
if /I "%~1"=="--help" (
  echo Usage: Start-MAVE-Workshop.cmd [--with-ur-rtde]
  exit /b 0
)
echo Unknown option: %~1
echo Usage: Start-MAVE-Workshop.cmd [--with-ur-rtde]
pause
exit /b 1

:args_done

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found on PATH.
  echo Install uv, then run this launcher again.
  pause
  exit /b 1
)

set "SYNC_EXTRA="
set "RUN_EXTRA="
if "%INSTALL_UR_RTDE%"=="1" set "SYNC_EXTRA= --extra robot"
if "%INSTALL_UR_RTDE%"=="1" set "RUN_EXTRA=--extra robot"
set "DID_SYNC=0"

if not exist "server\.venv\Scripts\python.exe" (
  echo Creating the server environment...
  pushd server
  if "%INSTALL_UR_RTDE%"=="1" echo Installing ur-rtde robot extra...
  uv sync%SYNC_EXTRA%
  if errorlevel 1 (
    popd
    echo uv sync failed.
    pause
    exit /b 1
  )
  popd
  set "DID_SYNC=1"
)

if "%DID_SYNC%"=="0" if exist "server\.venv\Scripts\python.exe" if "%INSTALL_UR_RTDE%"=="1" (
  echo Installing ur-rtde robot extra...
  pushd server
  uv sync --extra robot
  if errorlevel 1 (
    popd
    echo uv sync --extra robot failed.
    pause
    exit /b 1
  )
  popd
)

if "%INSTALL_UR_RTDE%"=="1" (
  echo Verifying ur-rtde Python bindings...
  pushd server
  uv run --extra robot python -c "import rtde_receive, rtde_control, rtde_io; print('ur-rtde Python bindings OK: import rtde_receive, rtde_control, rtde_io')"
  if errorlevel 1 (
    popd
    echo ur-rtde verification failed. The package is named ur-rtde, but its Python modules are rtde_receive, rtde_control, and rtde_io.
    pause
    exit /b 1
  )
  popd
)

pushd server
uv run %RUN_EXTRA% python -m operator_dashboard
set EXIT_CODE=%ERRORLEVEL%
popd

if not "%EXIT_CODE%"=="0" (
  echo Operator dashboard exited with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%

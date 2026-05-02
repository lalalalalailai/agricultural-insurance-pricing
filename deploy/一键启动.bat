@echo off
setlocal enabledelayedexpansion
chcp 936 >nul 2>&1
title Agri-Futures Pricing System

cd /d "%~dp0"

set LOG_DIR=%~dp0logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set LOG_FILE=%LOG_DIR%\startup_%date:~0,4%%date:~5,2%%date:~8,2%%time:~0,2%%time:~3,2%.log

(
echo ============================================================
echo   Agri-Futures Pricing System - Startup Log
echo   Time: %date% %time%
echo   Dir:  %CD%
echo   User: %USERNAME%
echo   PC:   %COMPUTERNAME%
echo ============================================================
) >> "%LOG_FILE%" 2>&1

echo.
echo   ========================================
echo     Agri-Futures Pricing System
echo   ========================================
echo.

call :log_info "Checking environment..."

set PYTHON_CMD=

for %%p in (python py python3) do (
    where %%p >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=%%p
        goto :python_found
    )
)

if exist "%~dp0..\.venv\Scripts\python.exe" (
    set PYTHON_CMD=%~dp0..\.venv\Scripts\python.exe
    call :log_info "Using venv Python"
    goto :python_found
)

call :log_error "Python NOT found in PATH or .venv!"
goto :error_exit

:python_found
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VER=%%i
call :log_info "Python found: %PY_VER%"

if not exist "src\app.py" (
    call :log_error "src\app.py NOT found"
    echo.
    echo   [ERROR] src\app.py missing!
    echo   Please run this bat from folder: 04_source_code
    goto :error_exit
)
call :log_info "Core file OK: src\app.py"

"%PYTHON_CMD%" -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :log_error "Streamlit not installed, installing..."
    echo.
    echo   [WARN] Installing Streamlit...
    pip install streamlit -q
    if %ERRORLEVEL% NEQ 0 (
        call :log_error "Streamlit install FAILED"
        goto :error_exit
    )
)
call :log_info "Streamlit dependency OK"

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    call :log_info "Port 8501 in use by PID %%a, killing..."
    taskkill /F /PID %%a >nul 2>&1
)
call :log_info "Port 8501 ready"

call :log_info "All checks passed, starting server..."
echo.
echo   [OK] WorkDir: %CD%
echo   [OK] Python:  %PYTHON_CMD%
echo   [OK] CoreFile: src\app.py
echo.
echo   Starting service...
echo   URL: http://localhost:8501
echo.
echo   Press Ctrl+C to stop
echo.

call :log_info "Running: %PYTHON_CMD% -m streamlit run src/app.py --server.port 8501 --browser.gatherUsageStats false --server.headless=false"
"%PYTHON_CMD%" -m streamlit run src/app.py --server.port 8501 --browser.gatherUsageStats false --server.headless=false

if %ERRORLEVEL% NEQ 0 (
    call :log_info "Server exited with code: %ERRORLEVEL%"
    goto :error_exit
)

call :log_info "Server stopped normally"
goto :normal_exit

:error_exit
echo.
echo   ========================================
echo   [ERROR] Startup FAILED!
echo   Log file: %LOG_FILE%
echo   ========================================
(
echo.
echo [ERROR] Startup FAILED - %date% %time%
echo ErrorLevel: %ERRORLEVEL%
) >> "%LOG_FILE%" 2>&1
echo.
echo   Press any key to exit...
pause >nul
exit /b 1

:normal_exit
echo.
echo   Server stopped. Log: %LOG_FILE%
echo   Press any key to exit...
pause >nul
exit /b 0

:log_info
echo [INFO] %~1
echo [%date% %time%] [INFO] %~1 >> "%LOG_FILE%" 2>&1
goto :eof

:log_error
echo [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%LOG_FILE%" 2>&1
goto :eof
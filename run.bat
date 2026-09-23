@echo off
title AlphaPredict.AI - Stock Price Prediction Server
cd /d "%~dp0"

echo ===================================================
echo   AlphaPredict.AI - ML Stock Price Prediction Engine
echo ===================================================
echo.

:: Detect Python executable
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PY_CMD=python
) else (
    if exist "C:\Users\gamef\AppData\Local\Python\bin\python.exe" (
        set PY_CMD="C:\Users\gamef\AppData\Local\Python\bin\python.exe"
    ) else (
        echo [ERROR] Python was not found in PATH or standard directory.
        pause
        exit /b 1
    )
)

echo Starting Flask server on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this window to stop the server.
echo.
start http://127.0.0.1:5000
%PY_CMD% app.py
pause

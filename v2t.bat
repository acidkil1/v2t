@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [X] venv not found. Run install.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    venv\Scripts\python.exe "%~dp0v2t.py" --help
    exit /b 0
)

venv\Scripts\python.exe "%~dp0v2t.py" %*
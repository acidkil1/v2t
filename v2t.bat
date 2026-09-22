@echo off
chcp 65001 >nul
setlocal

set "V2T_DIR=%~dp0"

if not exist "%V2T_DIR%venv\Scripts\python.exe" (
    echo [X] venv not found. Run install.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    "%V2T_DIR%venv\Scripts\python.exe" "%V2T_DIR%v2t.py" --help
    exit /b 0
)

"%V2T_DIR%venv\Scripts\python.exe" "%V2T_DIR%v2t.py" %*
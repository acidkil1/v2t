@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [X] venv not found. Run install.bat first.
    pause
    exit /b 1
)

echo Starting v2t GUI...
echo Browser will open at http://127.0.0.1:7860
echo To stop - close this window or press Ctrl+C.
echo.

venv\Scripts\python.exe "%~dp0app_gradio.py"

pause
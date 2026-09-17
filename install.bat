@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo === v2t installer ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python not found in PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     Do not forget "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

if exist venv (
    echo [--] venv already exists, skipping.
) else (
    echo [..] Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [X] Failed to create venv.
        pause
        exit /b 1
    )
    echo [OK] venv created.
)

echo [..] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo === Done ===
echo Run: v2t.bat "path\to\video.mp4"
echo      or start_gui.bat for GUI
pause
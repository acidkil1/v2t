@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: v2t "path\to\video.mp4" [--model small] [--lang ru] [--device auto]
    echo.
    echo Examples:
    echo   v2t "C:\video.mp4"
    echo   v2t "C:\video.mp4" --model medium --lang en
    echo   v2t "C:\video.mp4" --device cpu --delete-audio
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo [X] venv not found. Run install.bat first.
    pause
    exit /b 1
)

venv\Scripts\python.exe "%~dp0v2t.py" %*
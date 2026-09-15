@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo Использование: v2t "путь_к_видео" [--model small] [--lang ru] [--device auto]
    echo.
    echo Примеры:
    echo   v2t "C:\video.mp4"
    echo   v2t "C:\video.mp4" --model medium --lang en
    echo   v2t "C:\video.mp4" --device cpu --delete-audio
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo [X] Не найдено виртуальное окружение.
    echo     Сначала запусти install.bat
    pause
    exit /b 1
)

venv\Scripts\python.exe "%~dp0v2t.py" %*
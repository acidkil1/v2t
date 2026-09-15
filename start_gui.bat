@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [X] Не найдено виртуальное окружение.
    echo     Сначала запусти install.bat
    pause
    exit /b 1
)

echo Запуск v2t (GUI)...
echo Интерфейс откроется в браузере: http://127.0.0.1:7860
echo Для остановки закрой это окно или нажми Ctrl+C.
echo.

venv\Scripts\python.exe "%~dp0app_gradio.py"

pause
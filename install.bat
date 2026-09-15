@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo === Установка v2t ===
echo.

REM --- 1. Проверяем Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python не найден в PATH.
    echo     Установи Python 3.10 или 3.11 с https://www.python.org/downloads/
    echo     При установке отметь галочку "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

REM --- 2. Создаём venv ---
if exist venv (
    echo [--] Папка venv уже существует, пропускаю создание.
) else (
    echo [..] Создаю виртуальное окружение...
    python -m venv venv
    if errorlevel 1 (
        echo [X] Не удалось создать venv.
        pause
        exit /b 1
    )
    echo [OK] venv создан.
)

REM --- 3. Ставим зависимости ---
echo [..] Устанавливаю зависимости (это займёт пару минут)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Ошибка установки зависимостей.
    pause
    exit /b 1
)

echo.
echo === Готово ===
echo Запуск: v2t.bat "путь_к_видео.mp4"
pause
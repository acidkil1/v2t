@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo === Enabling GPU support (CUDA) ===
echo.

if not exist venv\Scripts\python.exe (
    echo [X] venv not found. Run install.bat first.
    pause
    exit /b 1
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [X] NVIDIA GPU not detected ^(nvidia-smi not found^).
    echo     GPU acceleration unavailable. CPU will be used.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected.
call venv\Scripts\activate.bat

echo [..] Installing CUDA libraries ^(~600 MB, may take a few minutes^)...
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
if errorlevel 1 (
    echo [X] Install failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [OK] GPU support enabled.
echo      Restart v2t / start_gui.bat to use CUDA.
pause
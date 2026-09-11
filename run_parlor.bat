@echo off
setlocal enabledelayedexpansion
title Parlor - Multimodal Voice Assistant (CUDA)

echo ===================================================
echo     Parlor - Multimodal AI (Voice + Vision)
echo             NVIDIA CUDA Acceleration
echo ===================================================
echo.
echo Select the Gemma backend model:
echo   [1] Gemma 4 E4B (Default - Fast, ~1.0-1.7s latency)
echo   [2] Gemma 4 12B (High Quality - Requires ~8GB VRAM)
echo.

set /p choice="Enter choice [1 or 2, default is 1]: "

if "%choice%"=="2" (
    set "MODEL=12b"
    echo.
    echo [*] Selected model: Gemma 4 12B
) else (
    set "MODEL=e4b"
    echo.
    echo [*] Selected model: Gemma 4 E4B
)

:: Set UTF-8 encoding for Python console I/O
set "PYTHONIOENCODING=utf-8"

:: Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [*] Starting Parlor server on http://localhost:8000 ...
python -m parlor.server

pause

@echo off
setlocal
cd /d "%~dp0"
title StudyRAG

REM ---- 1. Create .venv with whatever Python you have (3.10 - 3.15 all work) ----
if not exist ".venv\Scripts\activate.bat" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        py -m venv .venv
        if errorlevel 1 ( echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1 )
    )
)

REM ---- 2. Activate ----
call ".venv\Scripts\activate.bat"

REM ---- 3. Install packages once ----
if not exist ".venv\.installed" (
    echo Installing packages. This takes a few minutes the first time ...
    pip install -r requirements.txt
    if errorlevel 1 ( echo Install failed. & pause & exit /b 1 )
    echo done> ".venv\.installed"
)

REM ---- 4. NVIDIA API key (get one free at build.nvidia.com) ----
if "%NVIDIA_API_KEY%"=="" (
    if exist "nvidia_key.txt" (
        set /p NVIDIA_API_KEY=<nvidia_key.txt
    ) else (
        set /p NVIDIA_API_KEY=Paste your NVIDIA API key ^(nvapi-...^): 
        echo %NVIDIA_API_KEY%> nvidia_key.txt
    )
)

REM ---- 5. Run ----
echo.
echo Starting StudyRAG at http://localhost:8000  (Ctrl+C to stop)
start "" "http://localhost:8000"
uvicorn main:app --reload --port 8000

pause

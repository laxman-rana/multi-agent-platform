@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_DIR=.venv"
set "REQ_FILE=src\requirements.txt"

if not exist "%REQ_FILE%" (
    echo [ERROR] Requirements file not found: %REQ_FILE%
    exit /b 1
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Virtual environment already exists at %VENV_DIR%
) else (
    echo [INFO] Creating virtual environment at %VENV_DIR%
    py -3 -m venv "%VENV_DIR%" >nul 2>&1
    if errorlevel 1 (
        python -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment. Ensure Python is installed and on PATH.
            exit /b 1
        )
    )
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    exit /b 1
)

echo [INFO] Installing dependencies from %REQ_FILE%...
pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo.
echo [DONE] Environment is ready and active.
echo [TIP] Run this script with "call init.bat" to keep the environment activated in your current CMD session.

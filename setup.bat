@echo off
rem ======================================
rem  High-Fall Detection - Environment Setup
rem  Usage: Double-click to run
rem ======================================

echo ======================================
echo    High-Fall Detection - Setup
echo ======================================
echo.

rem ===== Step 1: Check Python =====
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found!
        echo Please install Python 3.8+ from: https://www.python.org/downloads/
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

echo OK: Python found (%PYTHON_CMD%)
echo.

rem ===== Step 2: Create venv =====
echo [2/6] Creating virtual environment...
if exist venv (
    echo venv/ already exists. Skipping creation.
) else (
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv!
        pause
        exit /b 1
    )
    echo OK: venv/ created
)
echo.

rem ===== Step 3: Activate venv =====
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv!
    pause
    exit /b 1
)
echo OK: venv activated
echo.

rem ===== Step 4: Upgrade pip =====
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo OK: pip upgraded
echo.

rem ===== Step 5: Install dependencies =====
echo [5/6] Installing dependencies (this may take a few minutes)...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
    echo OK: Dependencies installed
) else (
    echo WARNING: requirements.txt not found. Skipping.
)
echo.

rem ===== Step 6: Install Flask =====
echo [6/6] Installing Flask (Web UI)...
pip install flask >nul 2>&1
echo OK: Flask installed
echo.

rem ===== Done =====
echo ======================================
echo    Setup Complete!
echo ======================================
echo.
echo Next steps:
echo   1. Activate venc: venc\Scripts\activate.bat
echo   2. Start Web UI:  python -m src.web_app_launcher
echo   3. Open browser:   http://localhost:5000
echo.
pause

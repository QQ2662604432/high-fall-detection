@echo off
setlocal

rem Check if venv exists
if not exist venv (
    echo ========================================
    echo   ERROR: Virtual environment not found!
    echo ========================================
    echo.
    echo Please run setup.bat first:
    echo   setup.bat
    echo.
    echo This will install all dependencies.
    echo.
    pause
    exit /b 1
)

echo ========================================
echo   High-Fall Detection - Web UI
echo ========================================
echo.

rem Activate venv
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv!
    pause
    exit /b 1
)
echo OK: venv activated
echo.

rem Check Flask
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
)

rem Start Flask
echo Starting Web UI...
echo.
echo Open browser:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python -m src.web_app_launcher --host 0.0.0.0 --port 5000

pause

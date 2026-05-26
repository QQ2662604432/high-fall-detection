@echo off
rem =======================================
rem  High-Fall Detection - Web UI
rem =======================================
echo.
echo ======= Web UI Start =======
echo.

rem ======= Check Python =======
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found!
        echo Please install Python 3.8+
        echo https://www.python.org/downloads/
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

rem ======= Check venv =======
if not exist venv (
    echo ERROR: venv not found!
    echo Please run: setup.bat
    pause
    exit /b 1
)

rem ======= Activate venv =======
echo Activating venv...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv!
    pause
    exit /b 1
)
echo OK: venv activated
echo.

rem ======= Check Flask =======
%PYTHON_CMD% -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
    if errorlevel 1 (
        echo ERROR: Flask installation failed!
        pause
        exit /b 1
    )
)
echo OK: Flask ready
echo.

rem ======= Start Flask =======
echo =======================================
echo   Starting Web UI...
echo =======================================
echo.
echo [Local]  http://localhost:5000
echo [Local]  http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop
echo =======================================
echo.

python -m src.web_app_launcher --host 0.0.0.0 --port 5000

echo.
echo Process exited.
pause

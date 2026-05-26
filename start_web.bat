@echo off
chcp 65001 >nul 2>&1
echo =======================================
echo    High-Fall Detection - Web UI
echo =======================================
echo.

if not exist venv (
    echo ERROR: venv not found!
    echo Please run: setup.bat
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating venv...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv!
    pause
    exit /b 1
)
echo OK: venv activated
echo.

echo [2/3] Checking Flask...
python -c "import flask" >nul 2>&1
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

echo [3/3] Starting Web UI...
echo =======================================
echo    Web UI Starting...
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

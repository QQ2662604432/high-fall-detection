# ======================================
#  High-Fall Detection - PowerShell Web UI Startup
#  Usage: Right-click -> Run with PowerShell
# ======================================

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  High-Fall Detection - Web UI" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check venv
Write-Host "[1/4] Checking virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "ERROR: venv not found!" -ForegroundColor Red
    Write-Host "Please run: setup.ps1" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}
Write-Host "OK: venv/ found" -ForegroundColor Green
Write-Host ""

# Activate venv
Write-Host "[2/4] Activating venv..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate venv!" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "OK: venv activated" -ForegroundColor Green
Write-Host ""

# Check Flask
Write-Host "[3/4] Checking Flask..." -ForegroundColor Yellow
try {
    python -c "import flask" 2>&1 | Out-Null
    Write-Host "OK: Flask installed" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Flask not found, installing..." -ForegroundColor Yellow
    pip install flask
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Flask installation failed!" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "OK: Flask installed" -ForegroundColor Green
}
Write-Host ""

# Start Flask
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Starting Web UI..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Local access:" -ForegroundColor Green
Write-Host "  http://localhost:5000" -ForegroundColor White
Write-Host "  http://127.0.0.1:5000" -ForegroundColor White
Write-Host ""
Write-Host "LAN access (example):" -ForegroundColor Green
try {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" -ErrorAction SilentlyContinue).IPAddress
    if ($ip) {
        Write-Host "  http://$ip:5000" -ForegroundColor White
    }
} catch {}
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

python -m src.web_app_launcher --host 0.0.0.0 --port 5000 --debug

Write-Host ""
Write-Host "Process exited." -ForegroundColor Yellow
pause

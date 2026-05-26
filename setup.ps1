# ======================================
#  High-Fall Detection - PowerShell Setup
#  Usage: Right-click -> Run with PowerShell
# ======================================

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  High-Fall Detection - Environment Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "OK: Python found - $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# Step 2: Create venv
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "venv/ already exists. Skipping creation." -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create venv!" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "OK: venv/ created" -ForegroundColor Green
}
Write-Host ""

# Step 3: Activate venv and install dependencies
Write-Host "[3/5] Activating venv and upgrading pip..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip 2>&1 | Out-Null
Write-Host "OK: pip upgraded" -ForegroundColor Green
Write-Host ""

Write-Host "[4/5] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Dependency installation failed!" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "OK: Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found. Skipping." -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Install Flask
Write-Host "[5/5] Installing Flask (Web UI)..." -ForegroundColor Yellow
pip install flask
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Flask installation failed!" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "OK: Flask installed" -ForegroundColor Green
Write-Host ""

# Done
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start Web UI: .\start_web.ps1"
Write-Host "  2. Open browser: http://localhost:5000"
Write-Host ""
pause

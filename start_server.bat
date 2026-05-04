@echo off
REM =========================================
REM Insulator Detection Server Startup Script
REM =========================================

echo.
echo ========================================
echo  INSULATOR DETECTION SERVER LAUNCHER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Python detected successfully
echo.

REM Install/upgrade pip
echo [2/3] Installing required Python packages...
echo This may take a few minutes on first run...
echo.

python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install requirements!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [3/3] All packages installed successfully!
echo.

REM Check if uploads directory exists
if not exist "uploads" (
    echo Creating uploads directory...
    mkdir uploads
)

REM Check if model cache exists
if not exist "model_cache" (
    echo Creating model_cache directory...
    mkdir model_cache
)

echo ========================================
echo  STARTING FLASK SERVER
echo ========================================
echo.
echo Server will be available at:
echo   - Local:   http://127.0.0.1:5000
echo   - Network: http://192.168.246.206:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start Flask server
python app.py

REM If server stops, wait for user
echo.
echo Server stopped.
pause

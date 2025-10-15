@echo off
echo.
echo ========================================
echo   SQL Agent Flask Backend Startup
echo ========================================
echo.

REM Check if we're in the backend directory
if not exist "app.py" (
    echo Moving to backend directory...
    cd backend
)

if not exist "app.py" (
    echo ERROR: app.py not found!
    echo Please run this script from the backend directory.
    pause
    exit /b 1
)

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing Flask dependencies...
pip install flask flask-cors pandas langchain-openai langgraph

echo.
echo Starting Flask Backend Server...
echo.
echo ==========================================
echo   🌐 Server: http://localhost:5000
echo    Chat: POST to http://localhost:5000/chat
echo ==========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
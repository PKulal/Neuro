@echo off
REM Launches NeuroConnect AI using the project's virtual environment.
REM Double-click this instead of app.py -- the system Python does not have
REM TensorFlow, OpenCV or nibabel installed.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: .venv not found in "%CD%".
    echo Create it with:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 pause

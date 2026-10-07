@echo off
echo =====================================================
echo  GNSS Spoofing Detection - Start Backend (Python 3.12)
echo =====================================================
cd /d "%~dp0backend"

set VENV_DIR=venv312
if not exist %VENV_DIR%\Scripts\python.exe (
    set VENV_DIR=venv
)

echo Using Python environment: %VENV_DIR%

echo Generating synthetic dataset (if not exists)...
if not exist datasets\synthetic_gnss_dataset.csv (
    %VENV_DIR%\Scripts\python datasets\generate_synthetic.py
)

echo Creating demo user...
%VENV_DIR%\Scripts\python create_demo_user.py

echo.
echo Starting FastAPI backend on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
%VENV_DIR%\Scripts\python run.py

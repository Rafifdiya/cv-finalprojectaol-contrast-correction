@echo off
echo ================================================
echo  Automated Contrast Correction System - DEMO
echo  COMP7116001 Computer Vision - Team 2
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python tidak ditemukan. Install Python 3.10+ dulu.
    pause
    exit /b 1
)

REM Install dependencies if not yet installed
echo [1/2] Menginstall dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: Gagal install requirements. Cek koneksi internet.
    pause
    exit /b 1
)

echo.
echo [2/2] Menjalankan app...
echo.
echo App akan terbuka otomatis di browser.
echo Kalau tidak terbuka, buka manual: http://localhost:8501
echo.
echo Tekan Ctrl+C untuk stop app.
echo.

python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause

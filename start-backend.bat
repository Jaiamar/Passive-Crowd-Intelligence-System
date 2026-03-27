@echo off
echo ==========================================
echo  Passive Crowd Intelligence System
echo  Starting Backend (FastAPI + YOLO26)
echo ==========================================
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause

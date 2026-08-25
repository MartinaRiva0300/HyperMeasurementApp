@echo off
REM Forge 1GigE SWIR + TWINS hyperspectral acquisition
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --mode forge --fps 60
pause

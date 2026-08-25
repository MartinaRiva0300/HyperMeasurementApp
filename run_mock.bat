@echo off
REM No-hardware run: synthetic camera + "Simulate" on the TWINS stage panel
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --mode mock
pause

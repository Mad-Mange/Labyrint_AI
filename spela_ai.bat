@echo off
rem Sparad i teckentabell 850 (inte UTF-8) s† att †, „ och ” visas r„tt i cmd.
rem Dubbelklicka f”r att titta p† den f„rdigtr„nade AI:n (models\labyrint_ai.zip).
cd /d "%~dp0"
".venv\Scripts\python.exe" play.py --ai %*
if errorlevel 1 pause

@echo off
rem Sparad i teckentabell 850 (inte UTF-8) s† att †, „ och ” visas r„tt i cmd.
rem Startar en ny tr„ning och visar den live: spelet d„r AI:n ”var och ett f”nster med grafer.
rem Tr„nar p† banan med pinnar vid kanterna. St„ng det h„r f”nstret f”r att avbryta tr„ningen.
rem Extra flaggor skickas vidare till train.py, t.ex.  trana_ai_live.bat --steps 10e6
rem eller  trana_ai_live.bat --level classic  (den gamla banan utan pinnar).
cd /d "%~dp0"
for /f %%t in ('.venv\Scripts\python.exe -c "import time; print(time.strftime('%%Y%%m%%d-%%H%%M'))"') do set RUN=live-%%t
start "" ".venv\Scripts\pythonw.exe" play.py --live runs\%RUN%
start "" ".venv\Scripts\pythonw.exe" dashboard.py runs\%RUN%
title Labyrint - AI:n tr„nar (st„ng f”nstret f”r att avbryta)
".venv\Scripts\python.exe" train.py --name %RUN% --eval-freq 5e5 --level pinnar %*
echo.
pause

@echo off
setlocal
cd /d "D:\05_Quant\ML_4_Bot"
set PYTHONPATH=D:\05_Quant\ML_4_Bot
py -3.12 runners\check_fleet_status.py
pause
endlocal

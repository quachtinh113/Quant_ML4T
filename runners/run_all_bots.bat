@echo off
setlocal
cd /d "%~dp0.."
echo ======================================================
echo KICH HOAT CHU KY CHO TOAN BO 4 BOT EXNESS...
echo ======================================================

echo [1/4] Dang chay Bot BTC 8H...
call "%~dp0run_btc_8h.bat"

echo [2/4] Dang chay Bot FX D1...
call "%~dp0run_fx_d1.bat"

echo [3/4] Dang chay Bot USIDX Sess...
call "%~dp0run_usidx_sess.bat"

echo [4/4] Dang chay Bot Gold Sess...
call "%~dp0run_gold_sess.bat"

echo ======================================================
echo HOAN TAT! TONG HOP TRANG THAI FLEET:
echo ======================================================
py -3.12 runners\check_fleet_status.py

pause
endlocal

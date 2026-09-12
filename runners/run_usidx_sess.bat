@echo off
setlocal
cd /d "D:\05_Quant\ML_4_Bot"
set PYTHONPATH=D:\05_Quant\ML_4_Bot
set MT5_PATH=D:\05_Quant\ML_4_Bot\terminals\MT5_USIDX\terminal64.exe
set MT5_EXPECTED_LOGIN=463960820
set MT5_EXPECTED_SERVER=Exness-MT5Trial17
set LOG_FILE=D:\05_Quant\ML_4_Bot\logs\usidx_sess.log

echo ====================================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Running exness_usidx_sess (Account: 463960820) >> "%LOG_FILE%"
echo ====================================================== >> "%LOG_FILE%"

py -3.12 -m bots.exness_usidx_sess.deploy.deployment_loop >> "%LOG_FILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOG_FILE%"
endlocal

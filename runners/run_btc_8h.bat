@echo off
setlocal
cd /d "D:\05_Quant\ML_4_Bot"
set PYTHONPATH=D:\05_Quant\ML_4_Bot
set MT5_PATH=D:\05_Quant\ML_4_Bot\terminals\MT5_BTC\terminal64.exe
set MT5_EXPECTED_LOGIN=416351011
set MT5_EXPECTED_SERVER=Exness-MT5Trial14
set LOG_FILE=D:\05_Quant\ML_4_Bot\logs\btc_8h.log

echo ====================================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Running exness_btc_8h (Account: 416351011) >> "%LOG_FILE%"
echo ====================================================== >> "%LOG_FILE%"

py -3.12 -m bots.exness_btc_8h.deploy.deployment_loop --cycle >> "%LOG_FILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOG_FILE%"
endlocal

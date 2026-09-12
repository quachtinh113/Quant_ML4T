@echo off
setlocal
rem Builds dashboard\dashboard.html (once) and opens it in the default browser.
rem Never launches MT5, never places an order - see dashboard\README.md.
cd /d "%~dp0"

py -3.12 build_dashboard.py %*
if errorlevel 1 (
    echo dashboard: build failed, see the error above. >&2
    endlocal
    exit /b 1
)

if exist dashboard.html (
    start "" "dashboard.html"
) else (
    echo dashboard: build finished but dashboard.html was not found at %~dp0dashboard.html
)
endlocal

@echo off
rem Delegates to dashboard\run_dashboard.bat (same file, kept in both places per the
rem operator runbook's convention that every runner entry point lives in runners\).
rem Builds dashboard\dashboard.html (read-only) and opens it. Never launches MT5,
rem never places an order - see D:\05_Quant\ML_4_Bot\dashboard\README.md.
call "%~dp0..\dashboard\run_dashboard.bat" %*

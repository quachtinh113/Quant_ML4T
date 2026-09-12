# PowerShell script to register the 4 Exness Bot tasks into Windows Task Scheduler.
# Run from an elevated PowerShell (or a standard user: tasks then run only while logged in).
#
# TIME ZONES. Task Scheduler fires at the PC's local clock (Vietnam, UTC+7, no daylight saving).
#   * BTC 8h and FX D1 decide at fixed UTC hours, so one Vietnam time per decision is exact.
#   * Gold (London / New York session opens) and US indices (NYSE cash open / close) decide on
#     VENUE clocks that shift twice a year. Each of those legs is therefore registered at BOTH
#     its summer and its winter UTC hour, and the runner .bat calls runners\session_gate.py
#     first: the gate names the leg that is due at this hour, or exits 10 and the runner skips
#     the cycle. Off-hour firings cost one Python start and one "Skipped" log line, nothing else.
#     `py -3.12 runners\session_gate.py gold --table` prints this week's true decision hours.
#
# Every time below is Vietnam local; the UTC hour is in the comment. "+1 min" lets MT5 close
# the bar before the bot reads it.

$ErrorActionPreference = "Stop"

$workspace = "D:\05_Quant\ML_4_Bot"
$runners = Join-Path $workspace "runners"

$MonSun = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
$MonFri = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
$TueSat = @("Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")   # UTC Mon-Fri evening = VN Tue-Sat small hours

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Registering Exness 4-Bot Suite into Windows Task Scheduler" -ForegroundColor Cyan
Write-Host "Workspace: $workspace" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Each task: one action (the runner .bat, no arguments) and one trigger per (time, days) row.
$tasks = @(
    @{
        Name = "Exness_Bot_BTC_8h"
        Description = "Exness BTCUSD 8-hour decision cycle: 00:00, 08:00, 16:00 UTC (fixed) -> 07:01, 15:01, 23:01 VN, 7 days a week"
        Script = Join-Path $runners "run_btc_8h.bat"
        Triggers = @(
            @{ At = "07:01"; Days = $MonSun },   # 00:01 UTC
            @{ At = "15:01"; Days = $MonSun },   # 08:01 UTC
            @{ At = "23:01"; Days = $MonSun }    # 16:01 UTC
        )
    },
    @{
        Name = "Exness_Bot_FX_D1"
        Description = "Exness FX daily decision at 20:00 UTC (fixed) -> 03:01 VN Tue-Sat (= UTC Mon-Fri)"
        Script = Join-Path $runners "run_fx_d1.bat"
        Triggers = @(
            @{ At = "03:01"; Days = $TueSat }    # 20:01 UTC previous day
        )
    },
    @{
        Name = "Exness_Bot_USIDX_Sess"
        Description = "Exness US500/USTEC: NYSE cash open+30m (intraday) and cash close (overnight); summer AND winter UTC hours, runner gate picks the due leg"
        Script = Join-Path $runners "run_usidx_sess.bat"
        Triggers = @(
            @{ At = "21:01"; Days = $MonFri },   # 14:01 UTC intraday, US summer time (Mar-Nov)
            @{ At = "22:01"; Days = $MonFri },   # 15:01 UTC intraday, US winter time (Nov-Mar)
            @{ At = "03:01"; Days = $TueSat },   # 20:01 UTC overnight, US summer time
            @{ At = "04:01"; Days = $TueSat }    # 21:01 UTC overnight, US winter time
        )
    },
    @{
        Name = "Exness_Bot_Gold_Sess"
        Description = "Exness XAUUSD/XAGUSD: London open+1h and New York open+1h; summer AND winter UTC hours, runner gate picks the due leg"
        Script = Join-Path $runners "run_gold_sess.bat"
        Triggers = @(
            @{ At = "15:01"; Days = $MonFri },   # 08:01 UTC London leg, UK summer time (Mar-Oct)
            @{ At = "16:01"; Days = $MonFri },   # 09:01 UTC London leg, UK winter time (Oct-Mar)
            @{ At = "20:01"; Days = $MonFri },   # 13:01 UTC New York leg, US summer time
            @{ At = "21:01"; Days = $MonFri }    # 14:01 UTC New York leg, US winter time
        )
    }
)

foreach ($t in $tasks) {
    Write-Host "`nRegistering task: $($t.Name)..." -ForegroundColor Yellow

    $action = New-ScheduledTaskAction -Execute $t.Script -WorkingDirectory $workspace

    $triggers = @()
    foreach ($row in $t.Triggers) {
        $parsedTime = [datetime]::ParseExact($row.At, "HH:mm", $null)
        $triggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $row.Days -At $parsedTime
    }

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

    Register-ScheduledTask -TaskName $t.Name -Description $t.Description -Action $action -Trigger $triggers -Settings $settings -Force | Out-Null
    Write-Host " -> OK: $($t.Name) registered with $($triggers.Count) trigger(s)." -ForegroundColor Green
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "All 4 tasks have been configured in Task Scheduler." -ForegroundColor Green
Write-Host "Verify: Get-ScheduledTask Exness_Bot_* | Get-ScheduledTaskInfo" -ForegroundColor Cyan
Write-Host "Decision hours this week: py -3.12 runners\session_gate.py gold --table  (or usidx)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

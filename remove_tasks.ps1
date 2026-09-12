# PowerShell script to remove Exness Bot Tasks from Windows Task Scheduler

$ErrorActionPreference = "SilentlyContinue"

$tasks = @(
    "Exness_Bot_BTC_8h",
    "Exness_Bot_FX_D1",
    "Exness_Bot_USIDX_Sess",
    "Exness_Bot_Gold_Sess"
)

Write-Host "Unregistering Exness Bot tasks..." -ForegroundColor Yellow

foreach ($t in $tasks) {
    Unregister-ScheduledTask -TaskName $t -Confirm:$false
    Write-Host "Unregistered: $t" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Cyan

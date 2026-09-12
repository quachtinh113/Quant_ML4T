"""Fleet status monitor script (read-only: logs and the account halt file, nothing else).

It never imports MetaTrader5, never calls mt5.initialize (which launches a terminal when
none is running), and never reads .env. For the full picture use dashboard\build_dashboard.py.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path("D:/05_Quant/ML_4_Bot")
LOGS_DIR = BASE_DIR / "logs"
BOTS = ["btc_8h", "fx_d1", "usidx_sess", "gold_sess"]

print("=" * 65)
print(f"EXNESS 4-BOT FLEET STATUS - {datetime.now(timezone.utc).isoformat()}")
print("=" * 65)

# 1. Assigned Dedicated Accounts Overview
print("ASSIGNED DEDICATED ACCOUNTS:")
print("  - BTC_8H:     Terminal: MT5_BTC   | Login: 416351011 | Server: Exness-MT5Trial14")
print("  - FX_D1:      Terminal: MT5_FX    | Login: 463960816 | Server: Exness-MT5Trial17")
print("  - USIDX_SESS: Terminal: MT5_USIDX | Login: 463960820 | Server: Exness-MT5Trial17")
print("  - GOLD_SESS:  Terminal: MT5_GOLD  | Login: 463960823 | Server: Exness-MT5Trial17")

print("-" * 65)

# 2. Check Account Halt
halt_file = Path("D:/05_Quant/machine-learning-for-trading/data/mt5/monitor/account_halt.json")
if halt_file.exists():
    print("[WARNING] [ACCOUNT BREAKER HALT ACTIVE]:")
    try:
        data = json.loads(halt_file.read_text(encoding="utf-8"))
        print(f"   Reason: {data.get('reason')}")
        print(f"   Time:   {data.get('halted_at')}")
    except Exception as e:
        print(f"   (Failed to parse halt file: {e})")
else:
    print("[OK] [ACCOUNT BREAKER]: Normal (No halt file). All bots cleared.")

print("-" * 65)

# 2. Check each bot log
for bot in BOTS:
    log_file = LOGS_DIR / f"{bot}.log"
    if not log_file.exists():
        print(f"[{bot.upper()}]: Chưa có file log ({log_file.name})")
        continue

    lines = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    last_lines = lines[-5:] if len(lines) >= 5 else lines
    print(f"[{bot.upper()}] — {len(lines)} lines in log. Last activity:")
    for line in last_lines:
        print(f"   {line}")
    print()

print("=" * 65)
print("Full dashboard: runners\\run_dashboard.bat  (read-only, see dashboard\\README.md)")

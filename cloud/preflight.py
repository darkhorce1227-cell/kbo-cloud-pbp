from __future__ import annotations
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "state" / "status.json"
OUT = Path(os.environ["GITHUB_OUTPUT"])

def auto_target(now: datetime) -> date:
    return now.date() - timedelta(days=1) if now.hour < 6 else now.date()

target_raw = os.getenv("INPUT_TARGET_DATE", "").strip()
force = os.getenv("INPUT_FORCE", "false").lower() == "true"
target = date.fromisoformat(target_raw) if target_raw else auto_target(datetime.now(KST))

skip = False
reason = ""
if not force and STATUS.exists():
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
        if s.get("target_date") == target.isoformat() and s.get("validation") == "PASS":
            if s.get("state") in {"UPDATED", "ALREADY_CURRENT", "NO_GAMES"}:
                skip = True
                reason = f"Target {target} already complete: {s.get('state')}"
    except Exception:
        pass

with OUT.open("a", encoding="utf-8") as f:
    f.write(f"skip={'true' if skip else 'false'}\n")
    f.write(f"target_date={target.isoformat()}\n")
    f.write(f"reason={reason}\n")

print({"skip": skip, "target_date": target.isoformat(), "reason": reason})

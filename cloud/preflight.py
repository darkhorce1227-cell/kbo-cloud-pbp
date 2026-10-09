from __future__ import annotations
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "state" / "status.json"
AUDIT = ROOT / "state" / "postgame_score_audit.json"
OUT = Path(os.environ["GITHUB_OUTPUT"])


def auto_target(now: datetime) -> date:
    return now.date() - timedelta(days=1) if now.hour < 6 else now.date()


def audit_matches_status(status: dict, target: date) -> bool:
    if not AUDIT.exists():
        return False
    try:
        a = json.loads(AUDIT.read_text(encoding="utf-8"))
    except Exception:
        return False
    if a.get("target_date") != target.isoformat() or a.get("validation") != "PASS":
        return False
    status_sha = status.get("sha256_parquet")
    audit_sha = a.get("sha256_parquet")
    if status_sha and audit_sha and status_sha != audit_sha:
        return False
    return True


target_raw = os.getenv("INPUT_TARGET_DATE", "").strip()
force = os.getenv("INPUT_FORCE", "false").lower() == "true"
target = date.fromisoformat(target_raw) if target_raw else auto_target(datetime.now(KST))

skip = False
audit_backfill = False
reason = ""
if not force and STATUS.exists():
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
        if s.get("target_date") == target.isoformat() and s.get("validation") == "PASS":
            state = s.get("state")
            if state == "NO_GAMES":
                skip = True
                reason = f"Target {target} already complete: NO_GAMES"
            elif state in {"UPDATED", "ALREADY_CURRENT", "CONTINUITY_REPAIRED"}:
                if audit_matches_status(s, target):
                    skip = True
                    reason = f"Target {target} already complete with score audit: {state}"
                else:
                    audit_backfill = True
                    reason = f"Target {target} PBP is complete but score audit is missing/stale; force rebuild for audit backfill"
    except Exception:
        pass

with OUT.open("a", encoding="utf-8") as f:
    f.write(f"skip={'true' if skip else 'false'}\n")
    f.write(f"audit_backfill={'true' if audit_backfill else 'false'}\n")
    f.write(f"target_date={target.isoformat()}\n")
    f.write(f"reason={reason}\n")

print({"skip": skip, "audit_backfill": audit_backfill, "target_date": target.isoformat(), "reason": reason})

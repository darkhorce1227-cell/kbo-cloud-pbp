from __future__ import annotations
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
STATUS = STATE_DIR / "status.json"
AUDIT = STATE_DIR / "postgame_score_audit.json"
MANUAL_TRIGGER = STATE_DIR / "manual-trigger.txt"
OUT = Path(os.environ["GITHUB_OUTPUT"])
RECOVERY_LOOKBACK_DAYS = 7


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


def pending_prior_target(current_target: date) -> date | None:
    """Find the newest recent dated audit that had games but never passed.

    This is mainly for postseason/live-schedule lag: a later NO_GAMES/PASS status
    must not hide an unresolved previous-day slate.
    """
    for days in range(1, RECOVERY_LOOKBACK_DAYS + 1):
        candidate = current_target - timedelta(days=days)
        path = STATE_DIR / f"postgame_score_audit_{candidate.isoformat()}.json"
        if not path.exists():
            continue
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(audit.get("expected_games") or 0) <= 0:
            continue
        if audit.get("validation") != "PASS":
            return candidate
    return None


def manual_trigger_target_from_push() -> date | None:
    """Honor an explicit YYYY-MM-DD in manual-trigger.txt only on a push that changed that file.

    Scheduled runs and unrelated code pushes must not get pinned to an old manual target.
    """
    if os.getenv("GITHUB_EVENT_NAME") != "push":
        return None
    event_path = os.getenv("GITHUB_EVENT_PATH", "")
    if not event_path:
        return None
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    changed: set[str] = set()
    for commit in event.get("commits") or []:
        for key in ("added", "modified", "removed"):
            changed.update(str(p) for p in (commit.get(key) or []))
    if "state/manual-trigger.txt" not in changed or not MANUAL_TRIGGER.exists():
        return None
    try:
        text = MANUAL_TRIGGER.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(0))
    except ValueError:
        return None


target_raw = os.getenv("INPUT_TARGET_DATE", "").strip()
force = os.getenv("INPUT_FORCE", "false").lower() == "true"
automatic = auto_target(datetime.now(KST))
manual_target = None
if target_raw:
    target = date.fromisoformat(target_raw)
    recovery_target = None
else:
    manual_target = manual_trigger_target_from_push()
    if manual_target is not None:
        target = manual_target
        recovery_target = None
    else:
        recovery_target = pending_prior_target(automatic)
        target = recovery_target or automatic

skip = False
audit_backfill = False
reason = ""
if manual_target is not None:
    reason = f"Manual trigger requested explicit target {manual_target}"
elif recovery_target is not None and not force:
    # Deliberately do not let a later NO_GAMES/PASS status suppress this repair.
    reason = f"Recent unresolved score audit detected; recover {recovery_target} before {automatic}"
elif not force and STATUS.exists():
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

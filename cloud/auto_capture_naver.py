#!/usr/bin/env python3
"""Capture original Naver inning relays on a trusted machine with working Naver access.

Fail-closed: schedule/result/terminal/score/bytes must agree before writing a ZIP.
Never synthesizes PBP, never alters previously validated games.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "state/validated_game_pks_2026.json"
RAW_DIR = ROOT / "state/auto_pbp_raw"
KST = timezone(timedelta(hours=9))
SCHEDULE = "https://api-gw.sports.naver.com/schedule/games"
RELAY = "https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"
GAME_ID = re.compile(r"^[0-9]{8}[A-Z]{4}0[0-9]{4}$")
MAX_INNINGS = 15
MAX_DAYS_PER_RUN = 10


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_json(session: requests.Session, url: str, params: dict | None = None) -> tuple[bytes, dict]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=(10, 30))
            r.raise_for_status()
            content = r.content
            if not content or len(content) > 15_000_000:
                raise ValueError("Empty/oversized Naver response")
            obj = json.loads(content)
            if not isinstance(obj, dict) or obj.get("success") is not True:
                raise ValueError("Naver success != true")
            return content, obj
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt != 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Naver upstream failed: {url}; {type(error).__name__}: {error}")


def terminal(item: dict) -> bool:
    return any(str(opt.get("type")) == "99"
               for block in (item.get("textRelays") or [])
               for opt in (block.get("textOptions") or []))


def numeric_scoreboard(item: dict, game: dict, last_inning: int) -> tuple[dict, dict]:
    data = item.get("inningScore")
    if not isinstance(data, dict):
        raise ValueError("Missing inningScore")
    scores: dict[str, dict] = {}
    for side in ("away", "home"):
        values = data.get(side)
        if not isinstance(values, dict) or set(values) != {str(n) for n in range(1, last_inning + 1)}:
            raise ValueError(f"Incomplete inningScore {side}")
        if any(str(x) != "-" and not str(x).isdigit() for x in values.values()):
            raise ValueError(f"Invalid inningScore {side}")
        total = sum(int(x) for x in values.values() if str(x).isdigit())
        if total != int(game[f"{side}TeamScore"]):
            raise ValueError(f"Final score mismatch: {side} {total} vs schedule")
        scores[side] = values
    f5 = {}
    if last_inning >= 5:
        for side in ("away", "home"):
            first = [str(scores[side][str(n)]) for n in range(1, 6)]
            if any(not x.isdigit() for x in first):
                raise ValueError("Unplayed first-five inning")
            f5[side] = sum(int(x) for x in first)
    status = str(game.get("statusInfo") or "")
    ending = re.fullmatch(r"(\d+)회(초|말)", status)
    if not ending or int(ending.group(1)) != last_inning:
        raise ValueError(f"Schedule terminal inning mismatch: {status}")
    if str(scores["away"][str(last_inning)]) == "-":
        raise ValueError("Away terminal inning is unplayed")
    if ending.group(2) == "초" and str(scores["home"][str(last_inning)]) != "-":
        raise ValueError("Schedule says top-half finish but bottom is played")
    if ending.group(2) == "말" and str(scores["home"][str(last_inning)]) == "-":
        raise ValueError("Schedule says bottom-half finish but bottom unplayed")
    return scores, f5


def capture_one(session: requests.Session, date_s: str, original_schedule: bytes, game: dict) -> Path:
    gid = str(game["gameId"])
    if not GAME_ID.fullmatch(gid) or gid[4:8] != date_s[5:7] + date_s[8:10] or not gid.endswith(date_s[:4]):
        raise ValueError(f"Invalid Naver game ID/date: {gid}")
    if str(game.get("statusCode")) != "RESULT" or int(game.get("statusNum", -1)) != 4:
        raise ValueError(f"Game is not final: {gid}")
    away, home = str(game.get("awayTeamCode")), str(game.get("homeTeamCode"))
    if gid[8:10] != away or gid[10:12] != home:
        raise ValueError(f"Team codes disagree with game PK: {gid}")
    files: dict[str, bytes] = {"schedule.json": original_schedule}
    innings = []
    result = None
    for number in range(1, MAX_INNINGS + 1):
        name = f"relays/inning_{number:02d}.json"
        raw, wrapper = request_json(session, RELAY.format(game_id=gid), {"inning": number})
        item = (wrapper.get("result") or {}).get("textRelayData")
        if not isinstance(item, dict) or str(item.get("gameId")) != gid:
            raise ValueError(f"Wrong/empty Naver relay: {gid} inning {number}")
        if not isinstance(item.get("textRelays"), list):
            raise ValueError(f"Missing textRelays: {gid} inning {number}")
        end = terminal(item)
        files[name] = raw
        innings.append({"inning": number, "file": name, "bytes": len(raw),
                        "sha256": sha(raw), "game_end_type_99": end})
        print(f"  {gid} inning {number}: {len(raw)} bytes" + (" [GAME_END]" if end else ""), flush=True)
        if end:
            result = item
            break
    if result is None:
        raise ValueError(f"No explicit GAME_END through inning {MAX_INNINGS}: {gid}")
    scores, f5 = numeric_scoreboard(result, game, len(innings))
    manifest = {
        "schema": "kbo_naver_raw_relay_capture_v1",
        "target_date_kst": date_s,
        "game_id": gid,
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inning_count": len(innings),
        "schedule_sha256": sha(original_schedule),
        "game_end_verified_in_relay": True,
        "innings": innings,
    }
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    dest = RAW_DIR / f"kbo_{date_s.replace('-', '')}_{away}{home}_raw_{timestamp}.zip"
    if dest.exists():
        raise FileExistsError(f"Archive already exists: {dest}")
    tmp = dest.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, content in files.items():
                z.writestr(name, content)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None or set(z.namelist()) != set(files):
                raise ValueError("Corrupt or incomplete ZIP")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"CAPTURED {gid} final {scores['away']} / {scores['home']} F5={f5} ZIP_SHA256={sha(dest.read_bytes())}", flush=True)
    return dest


def run(start_override: str | None, end_override: str | None) -> int:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if int(index.get("year", 0)) != 2026:
        raise RuntimeError("2026 baseline index missing or incorrect; no blind bootstrap")
    already_validated = set(str(x) for x in index["game_pks"])
    if len(already_validated) != int(index["game_count"]):
        raise RuntimeError("Baseline index has duplicate/mismatched game IDs")
    start = date.fromisoformat(start_override) if start_override else date.fromisoformat(index["through_date"]) + timedelta(days=1)
    end = date.fromisoformat(end_override) if end_override else datetime.now(KST).date() - timedelta(days=1)
    if end < start:
        print(f"NO_DATES: through={index['through_date']} start={start} end={end}")
        return 0
    end = min(end, start + timedelta(days=MAX_DAYS_PER_RUN - 1))
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://m.sports.naver.com",
        "Referer": "https://m.sports.naver.com/",
    })
    captured = []
    for offset in range((end - start).days + 1):
        current = start + timedelta(days=offset)
        d = current.isoformat()
        original, response = request_json(session, SCHEDULE, {
            "fields": "basic,categoryName,statusNum", "upperCategoryId": "kbaseball",
            "fromDate": d, "toDate": d,
        })
        games = (response.get("result") or {}).get("games")
        if not isinstance(games, list):
            raise ValueError(f"Missing Naver schedule result.games: {d}")
        real = [g for g in games
                if isinstance(g, dict) and g.get("gameDate") == d
                and GAME_ID.fullmatch(str(g.get("gameId", "")))
                and str(g["gameId"]).endswith("2026")]
        seen = set()
        for game in real:
            gid = str(game["gameId"])
            if gid in seen:
                raise ValueError(f"Duplicate Naver schedule entry: {gid}")
            seen.add(gid)
            if str(game.get("statusCode")) != "RESULT" or int(game.get("statusNum", -1)) != 4:
                print(f"NOT_FINAL {d} {gid}; try next scheduled run")
                continue
            if gid in already_validated:
                continue
            prefix = f"kbo_{d.replace('-', '')}_{game['awayTeamCode']}{game['homeTeamCode']}_raw_"
            existing = list(RAW_DIR.glob(prefix + "*.zip")) if RAW_DIR.exists() else []
            if existing:
                print(f"ALREADY_CAPTURED {gid} ({len(existing)} archive(s)); awaiting GitHub import")
                continue
            captured.append(capture_one(session, d, original, game))
        print(f"SCHEDULE_OK {d}: {len(real)} real KBO entries", flush=True)
    print(f"CAPTURE_COMPLETE count={len(captured)} from={start} through={end}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", help="Override start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Override end date YYYY-MM-DD")
    args = parser.parse_args()
    try:
        return run(args.start_date, args.end_date)
    except Exception as exc:
        print(f"CAPTURE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

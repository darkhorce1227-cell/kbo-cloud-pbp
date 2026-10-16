#!/usr/bin/env python3
"""Probe whether Naver season/schedule endpoints can reproduce RS starter-average IP cheaply."""
import json
from kbo_pbp import naver, schedule


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            yield p, v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from walk(v, p)


def main():
    year = 2025
    raw = naver.fetch_schedule(year)
    rs = [g for g in raw if g.get("roundCode") == schedule.REGULAR]
    if not rs:
        raise SystemExit("no regular-season games")
    game = rs[0]
    print("SCHEDULE_GAME_ID", game.get("gameId") or game.get("id"))
    print("SCHEDULE_TOP_KEYS", json.dumps(sorted(game.keys()), ensure_ascii=False))

    candidates = []
    for path, value in walk(game):
        low = path.lower()
        if any(tok in low for tok in ("pitcher", "pcode", "playercode", "playerid")):
            print("SCHEDULE_CANDIDATE", path, json.dumps(value, ensure_ascii=False, default=str))
            if isinstance(value, (str, int)) and str(value).strip():
                candidates.append(str(value).strip())

    tried = []
    for pcode in candidates:
        if pcode in tried:
            continue
        tried.append(pcode)
        try:
            result = naver.fetch_player_season(pcode, year)
        except Exception as exc:
            print("PLAYER_FETCH_FAIL", pcode, type(exc).__name__, str(exc))
            continue
        ps = result.get("pitcherStats")
        if ps:
            print("PLAYER_CODE", pcode)
            print("PITCHER_STATS", json.dumps(ps, ensure_ascii=False, sort_keys=True, default=str))
            return
    print("NO_FETCHABLE_STARTER_CODE_FROM_SCHEDULE")


if __name__ == "__main__":
    main()

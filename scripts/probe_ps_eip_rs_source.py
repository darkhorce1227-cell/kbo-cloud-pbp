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
        for i, v in enumerate(obj[:20]):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from walk(v, p)


def scalar_player_candidates(obj):
    out = []
    for path, value in walk(obj):
        low = path.lower()
        if any(tok in low for tok in ("pcode", "playercode", "playerid", "pitchercode", "pitcherid")):
            if isinstance(value, (str, int)) and str(value).strip():
                out.append((path, str(value).strip()))
    return out


def main():
    year = 2025
    raw = naver.fetch_schedule(year)
    rs = [g for g in raw if g.get("roundCode") == schedule.REGULAR]
    if not rs:
        raise SystemExit("no regular-season games")
    game = rs[0]
    gid = game.get("gameId") or game.get("id")
    print("SCHEDULE_GAME_ID", gid)
    print("AWAY_STARTER_NAME", game.get("awayStarterName"))
    print("HOME_STARTER_NAME", game.get("homeStarterName"))

    relay = naver.fetch_relay(str(gid), 1)
    print("RELAY_TOP_KEYS", json.dumps(sorted(relay.keys()), ensure_ascii=False))

    candidates = scalar_player_candidates(relay)
    for path, value in candidates[:50]:
        print("RELAY_PLAYER_CANDIDATE", path, value)

    tried = []
    for path, pcode in candidates:
        if pcode in tried:
            continue
        tried.append(pcode)
        try:
            result = naver.fetch_player_season(pcode, year)
        except Exception as exc:
            continue
        ps = result.get("pitcherStats")
        if ps:
            player = result.get("player") or {}
            print("PLAYER_CODE", pcode)
            print("PLAYER_NAME", player.get("name") or player.get("playerName") or player.get("nameKo"))
            print("PLAYER_SOURCE_PATH", path)
            print("PITCHER_STATS", json.dumps(ps, ensure_ascii=False, sort_keys=True, default=str))
            # keep probing until both schedule starters are found if names are available
            found_name = str(player.get("name") or player.get("playerName") or player.get("nameKo") or "")
            if found_name in {str(game.get("awayStarterName") or ""), str(game.get("homeStarterName") or "")}:
                print("MATCHED_SCHEDULE_STARTER", found_name)
                return
    print("NO_FETCHABLE_STARTER_CODE_FROM_RELAY")


if __name__ == "__main__":
    main()

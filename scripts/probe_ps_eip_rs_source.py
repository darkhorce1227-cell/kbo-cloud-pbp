#!/usr/bin/env python3
"""Probe whether one inning-one Naver relay contains enough data to rebuild RS starter game logs."""
import json
from kbo_pbp import naver, schedule


def main():
    year = 2025
    raw = naver.fetch_schedule(year)
    rs = [g for g in raw if g.get("roundCode") == schedule.REGULAR]
    if not rs:
        raise SystemExit("no regular-season games")
    game = rs[0]
    gid = str(game.get("gameId") or game.get("id"))
    relay = naver.fetch_relay(gid, 1)

    print("GAME", gid, game.get("awayTeamName"), "@", game.get("homeTeamName"))
    for side in ("away", "home"):
        lineup = relay.get(f"{side}Lineup") or {}
        entry = relay.get(f"{side}Entry") or {}
        lp = lineup.get("pitcher") or []
        ep = entry.get("pitcher") or []
        print(side.upper(), "LINEUP_PITCHER0", json.dumps(lp[0] if lp else None, ensure_ascii=False, sort_keys=True, default=str))
        print(side.upper(), "ENTRY_PITCHERS", json.dumps(ep, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

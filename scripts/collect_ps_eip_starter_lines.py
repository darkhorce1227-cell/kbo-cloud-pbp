#!/usr/bin/env python3
"""Collect regular-season and postseason starter game lines from one Naver relay call/game.

Naver's inning-1 relay payload exposes the completed game pitching line in
{away,home}Lineup.pitcher. The seqno=1 pitcher is the starter and includes
`inn`, `ballCount`, R/ER, etc. This lets us rebuild starter-only RS average IP
without downloading every inning/pitch of the regular season.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from kbo_pbp import naver, schedule


def parse_ip(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if " " in s and "/" in s:
        whole, frac = s.split(None, 1)
        num, den = frac.split("/", 1)
        return float(whole) + float(num) / float(den)
    if "." in s:
        whole, frac = s.split(".", 1)
        if frac in {"0", "1", "2"}:
            return float(whole) + int(frac) / 3.0
    return float(s)


def _unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def candidate_game_ids(game, year, gid):
    date = str(game.get("gameDate") or "")[:10].replace("-", "")
    away = str(game.get("awayTeamCode") or "").strip()
    home = str(game.get("homeTeamCode") or "").strip()
    year_s = str(year)
    candidates = [gid]
    if gid and not gid.endswith(year_s):
        candidates.append(f"{gid}{year_s}")
    if len(date) == 8 and away and home:
        for header in ("0", "1", "2"):
            candidates.append(f"{date}{away}{home}{header}{year_s}")
        candidates.append(f"{date}{away}{home}0")
    return _unique(candidates)


def fetch_first_relay(game, year, pause):
    gid = str(game.get("gameId") or game.get("id") or "")
    errors = []
    for candidate in candidate_game_ids(game, year, gid):
        try:
            relay = naver.fetch_relay(candidate, 1)
            if pause:
                time.sleep(pause)
            if relay.get("awayLineup") and relay.get("homeLineup"):
                return candidate, relay
            errors.append(f"{candidate}: missing lineup")
        except Exception as exc:
            if pause:
                time.sleep(pause)
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors[:6]))


def starter_from_relay(relay, side):
    pitchers = ((relay.get(f"{side}Lineup") or {}).get("pitcher") or [])
    if not pitchers:
        raise ValueError(f"{side} lineup has no pitchers")
    ordered = sorted(pitchers, key=lambda p: int(p.get("seqno") or 999))
    starter = ordered[0]
    if int(starter.get("seqno") or 1) != 1:
        raise ValueError(f"{side} first pitcher seqno != 1")
    ip = parse_ip(starter.get("inn"))
    if ip is None:
        raise ValueError(f"{side} starter missing inn")
    return starter, ip


def collect_round(year, games, round_group, pause):
    rows, errors = [], []
    for idx, (_, game_s) in enumerate(games.iterrows(), start=1):
        game = game_s.to_dict()
        gid = str(game.get("gameId") or game.get("id") or "")
        try:
            relay_gid, relay = fetch_first_relay(game, year, pause)
            for side in ("away", "home"):
                starter, ip = starter_from_relay(relay, side)
                rows.append({
                    "season": year,
                    "round_group": round_group,
                    "round_code": game.get("roundCode"),
                    "game_id": gid,
                    "relay_game_id": relay_gid,
                    "game_date": str(game.get("gameDate") or "")[:10],
                    "away_team_code": game.get("awayTeamCode"),
                    "home_team_code": game.get("homeTeamCode"),
                    "side": side,
                    "starter_pcode": str(starter.get("pcode") or ""),
                    "starter_name": starter.get("name"),
                    "seqno": starter.get("seqno"),
                    "actual_ip": ip,
                    "inn_raw": starter.get("inn"),
                    "pitch_count": starter.get("ballCount"),
                    "runs": starter.get("run"),
                    "earned_runs": starter.get("er"),
                    "hits": starter.get("hit"),
                    "walks": starter.get("bb"),
                    "hbp": starter.get("hbp"),
                    "strikeouts": starter.get("kk"),
                    "home_runs": starter.get("hr"),
                })
        except Exception as exc:
            errors.append({"game_id": gid, "error": f"{type(exc).__name__}: {exc}"})
        if idx % 100 == 0:
            print(json.dumps({"year": year, "round": round_group, "done_games": idx, "rows": len(rows), "errors": len(errors)}), flush=True)
    return pd.DataFrame(rows), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--output-dir", default="research/postseason-eip/source")
    ap.add_argument("--pause", type=float, default=0.12)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    season_df = schedule.fetch(args.year)
    rs_games = schedule.playable(season_df, rounds=(schedule.REGULAR,))
    ps_games = schedule.playable(season_df, rounds=schedule.POSTSEASON)

    rs, rs_errors = collect_round(args.year, rs_games, "RS", args.pause)
    ps, ps_errors = collect_round(args.year, ps_games, "PS", args.pause)

    rs_path = out / f"{args.year}_rs_starters.csv"
    ps_path = out / f"{args.year}_ps_starters.csv"
    err_path = out / f"{args.year}_errors.json"
    rs.to_csv(rs_path, index=False)
    ps.to_csv(ps_path, index=False)
    err_path.write_text(json.dumps({"rs": rs_errors, "ps": ps_errors}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "year": args.year,
        "rs_games_expected": int(len(rs_games)),
        "rs_games_collected": int(rs["game_id"].nunique()) if not rs.empty else 0,
        "rs_starter_rows": int(len(rs)),
        "ps_games_expected": int(len(ps_games)),
        "ps_games_collected": int(ps["game_id"].nunique()) if not ps.empty else 0,
        "ps_starter_rows": int(len(ps)),
        "rs_errors": len(rs_errors),
        "ps_errors": len(ps_errors),
    }
    (out / f"{args.year}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if rs_errors or ps_errors or summary["rs_starter_rows"] != 2 * summary["rs_games_expected"] or summary["ps_starter_rows"] != 2 * summary["ps_games_expected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

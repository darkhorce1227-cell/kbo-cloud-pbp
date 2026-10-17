#!/usr/bin/env python3
"""Collect one completed regular-season starter pitching line per game-side.

Naver's inning-1 relay payload exposes the completed game pitching lines in
{away,home}Lineup.pitcher. The pitcher with seqno=1 is the starter and `inn`
is his completed-game innings value. This avoids downloading pitch-level data
for every regular-season game while preserving Naver's stable player pcode.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from kbo_pbp import naver, schedule


def parse_ip(value) -> tuple[int, float]:
    """Return (outs, decimal baseball IP) from Naver innings text."""
    if value is None:
        raise ValueError("missing inn")
    s = str(value).strip()
    if not s:
        raise ValueError("blank inn")

    if " " in s and "/" in s:
        whole, frac = s.split(None, 1)
        num, den = frac.split("/", 1)
        whole_i, num_i, den_i = int(whole), int(num), int(den)
        if den_i != 3 or num_i not in (0, 1, 2):
            raise ValueError(f"unsupported innings fraction: {s}")
        outs = 3 * whole_i + num_i
        return outs, outs / 3.0

    if "." in s:
        whole, frac = s.split(".", 1)
        if frac in {"0", "1", "2"}:
            outs = 3 * int(whole) + int(frac)
            return outs, outs / 3.0
        raise ValueError(f"unsupported decimal innings encoding: {s}")

    # A bare integer means whole innings.
    outs = 3 * int(s)
    return outs, outs / 3.0


def _unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def candidate_game_ids(game: dict, year: int, gid: str) -> list[str]:
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


def fetch_first_relay(game: dict, year: int, pause: float, retries: int):
    gid = str(game.get("gameId") or game.get("id") or "")
    errors = []
    for candidate in candidate_game_ids(game, year, gid):
        for attempt in range(1, retries + 1):
            try:
                relay = naver.fetch_relay(candidate, 1)
                if relay.get("awayLineup") and relay.get("homeLineup"):
                    if pause:
                        time.sleep(pause)
                    return candidate, relay
                errors.append(f"{candidate}: missing lineup")
                break
            except Exception as exc:  # network/API failures are retried
                errors.append(
                    f"{candidate} attempt {attempt}: {type(exc).__name__}: {exc}"
                )
                if pause:
                    time.sleep(max(pause, 0.1) * attempt)
    raise RuntimeError("; ".join(errors[-10:]))


def starter_from_relay(relay: dict, side: str):
    pitchers = ((relay.get(f"{side}Lineup") or {}).get("pitcher") or [])
    if not pitchers:
        raise ValueError(f"{side} lineup has no pitchers")

    def seq(p):
        try:
            return int(p.get("seqno"))
        except Exception:
            return 999

    ordered = sorted(pitchers, key=seq)
    starter = ordered[0]
    if seq(starter) != 1:
        raise ValueError(f"{side} first pitcher seqno != 1")
    pcode = str(starter.get("pcode") or "").strip()
    if not pcode:
        raise ValueError(f"{side} starter missing pcode")
    outs, ip = parse_ip(starter.get("inn"))
    return starter, outs, ip


def collect_year(year: int, pause: float, retries: int):
    season_df = schedule.fetch(year)
    games = schedule.playable(season_df, rounds=(schedule.REGULAR,))
    rows, errors = [], []

    for idx, (_, game_s) in enumerate(games.iterrows(), start=1):
        game = game_s.to_dict()
        gid = str(game.get("gameId") or game.get("id") or "")
        try:
            relay_gid, relay = fetch_first_relay(game, year, pause, retries)
            for side in ("away", "home"):
                starter, outs, ip = starter_from_relay(relay, side)
                team_code = (
                    game.get("awayTeamCode") if side == "away" else game.get("homeTeamCode")
                )
                opponent_code = (
                    game.get("homeTeamCode") if side == "away" else game.get("awayTeamCode")
                )
                rows.append(
                    {
                        "season": year,
                        "round_code": game.get("roundCode"),
                        "game_id": gid,
                        "relay_game_id": relay_gid,
                        "game_date": str(game.get("gameDate") or "")[:10],
                        "away_team_code": game.get("awayTeamCode"),
                        "home_team_code": game.get("homeTeamCode"),
                        "side": side,
                        "team_code": team_code,
                        "opponent_code": opponent_code,
                        "starter_pcode": str(starter.get("pcode") or "").strip(),
                        "starter_name": starter.get("name"),
                        "seqno": int(starter.get("seqno") or 1),
                        "actual_outs": outs,
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
                        "source": "naver_inning1_completed_lineup_pitcher",
                    }
                )
        except Exception as exc:
            errors.append({"game_id": gid, "error": f"{type(exc).__name__}: {exc}"})

        if idx % 100 == 0:
            print(
                json.dumps(
                    {
                        "year": year,
                        "done_games": idx,
                        "expected_games": int(len(games)),
                        "rows": len(rows),
                        "errors": len(errors),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    return games, frame, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--output-dir", default="historical_rs/source")
    ap.add_argument("--pause", type=float, default=0.03)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    games, rows, errors = collect_year(args.year, args.pause, args.retries)

    csv_path = out / f"{args.year}_rs_starters.csv"
    err_path = out / f"{args.year}_rs_errors.json"
    summary_path = out / f"{args.year}_rs_summary.json"
    rows.to_csv(csv_path, index=False)
    err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    games_expected = int(len(games))
    games_collected = int(rows["game_id"].nunique()) if not rows.empty else 0
    starter_rows = int(len(rows))
    duplicate_game_sides = (
        int(rows.duplicated(["season", "game_id", "side"], keep=False).sum())
        if not rows.empty
        else 0
    )
    bad_thirds = (
        int(((rows["actual_ip"] * 3 - rows["actual_outs"]).abs() > 1e-10).sum())
        if not rows.empty
        else 0
    )
    summary = {
        "year": args.year,
        "status": "PASS",
        "rs_games_expected": games_expected,
        "rs_games_collected": games_collected,
        "rs_starter_rows": starter_rows,
        "errors": len(errors),
        "duplicate_game_sides": duplicate_game_sides,
        "bad_ip_thirds": bad_thirds,
    }
    if (
        errors
        or games_collected != games_expected
        or starter_rows != 2 * games_expected
        or duplicate_game_sides
        or bad_thirds
    ):
        summary["status"] = "FAIL"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

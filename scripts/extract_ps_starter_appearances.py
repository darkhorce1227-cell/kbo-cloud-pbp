#!/usr/bin/env python3
"""Extract postseason starter appearances and actual IP from pitch-level PBP.

This is a reconstruction utility only. It does not calculate the historical
Expected-IP shift. It converts an already-built postseason pitch-level master
into one row per starting pitcher/game with exact recorded outs inferred from
PA start/end out states.

Starter rule
------------
For each game/defensive side, the first pitcher observed is the starter:
- TOP half => home team is pitching
- BOT half => away team is pitching

Outs rule
---------
Pitch rows repeat PA-level end-state fields. We collapse to one row per PA and
assign that PA's recorded outs to the pitcher on the final row of the PA. This
handles double/triple plays and the rare PA that closes on a baserunning third
out. Actual IP is recorded_outs / 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "season",
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "pitcher_name",
    "outs_when_up",
    "post_outs",
}


def _is_top(value: object) -> bool:
    s = str(value).strip().lower()
    return s.startswith("top") or s in {"초", "t"}


def _is_bottom(value: object) -> bool:
    s = str(value).strip().lower()
    return s.startswith("bot") or s.startswith("bottom") or s in {"말", "b"}


def _defense_side(value: object) -> str:
    if _is_top(value):
        return "home"
    if _is_bottom(value):
        return "away"
    raise ValueError(f"Unrecognized inning_topbot value: {value!r}")


def _prepare_pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    missing = sorted(REQUIRED - set(x.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    for col in ("outs_when_up", "post_outs", "inning", "at_bat_number", "pitch_number"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[["outs_when_up", "post_outs", "inning", "at_bat_number", "pitch_number"]].isna().any().any():
        raise RuntimeError("NA/non-numeric values in inning/PA/out state columns")

    x["defense_side"] = x["inning_topbot"].map(_defense_side)
    x["pitcher"] = x["pitcher"].astype(str)

    sort_cols = ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"]
    x = x.sort_values(sort_cols, kind="stable")

    # One row per PA, taking the final observed pitcher so mid-PA pitching
    # changes credit subsequent outs to the pitcher who actually finished it.
    pa = x.groupby(["game_pk", "at_bat_number"], as_index=False, sort=False).tail(1).copy()
    pa["outs_recorded"] = pa["post_outs"].astype(int) - pa["outs_when_up"].astype(int)

    bad = pa.loc[(pa["outs_recorded"] < 0) | (pa["outs_recorded"] > 3)]
    if not bad.empty:
        sample = bad[["game_pk", "inning", "inning_topbot", "at_bat_number", "outs_when_up", "post_outs"]].head(10)
        raise RuntimeError(f"invalid PA outs delta; sample=\n{sample.to_string(index=False)}")

    return pa


def extract_starter_appearances(df: pd.DataFrame) -> pd.DataFrame:
    pa = _prepare_pa_rows(df)

    # Determine starter from the first chronological pitcher observed on each
    # defensive side. Use the full pitch stream rather than PA finals so a rare
    # mid-PA replacement cannot erase the true starter identity.
    x = df.copy()
    x["defense_side"] = x["inning_topbot"].map(_defense_side)
    x["pitcher"] = x["pitcher"].astype(str)
    x = x.sort_values(
        ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
        kind="stable",
    )
    first = (
        x.groupby(["game_pk", "defense_side"], as_index=False, sort=False)
        .first()[["game_pk", "defense_side", "pitcher", "pitcher_name"]]
        .rename(columns={"pitcher": "starter_id", "pitcher_name": "starter_name"})
    )

    # Aggregate outs charged to every pitcher from PA-final ownership, then keep
    # only the identified starter for each side.
    pitcher_outs = (
        pa.groupby(["game_pk", "defense_side", "pitcher"], as_index=False)["outs_recorded"]
        .sum()
        .rename(columns={"pitcher": "starter_id", "outs_recorded": "recorded_outs"})
    )
    out = first.merge(pitcher_outs, on=["game_pk", "defense_side", "starter_id"], how="left")
    out["recorded_outs"] = out["recorded_outs"].fillna(0).astype(int)
    out["actual_ip"] = out["recorded_outs"] / 3.0

    # Attach game metadata from the first row of each game.
    meta_cols = [
        c for c in [
            "season", "game_pk", "game_date", "home_team", "away_team",
            "round_code", "data_quality", "coverage_status",
        ] if c in df.columns
    ]
    meta = df.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"], kind="stable")
    meta = meta.groupby("game_pk", as_index=False, sort=False).first()[meta_cols]
    out = out.merge(meta, on="game_pk", how="left")

    out["team"] = out.apply(
        lambda r: r.get("home_team") if r["defense_side"] == "home" else r.get("away_team"),
        axis=1,
    )
    out["opponent"] = out.apply(
        lambda r: r.get("away_team") if r["defense_side"] == "home" else r.get("home_team"),
        axis=1,
    )

    # Hard integrity check: complete games should yield exactly two starters.
    counts = out.groupby("game_pk").size()
    bad_games = counts[counts != 2]
    if not bad_games.empty:
        raise RuntimeError(f"games without exactly two identified starters: {bad_games.to_dict()}")

    preferred = [
        "season", "game_pk", "game_date", "defense_side", "team", "opponent",
        "starter_id", "starter_name", "recorded_outs", "actual_ip",
        "round_code", "data_quality", "coverage_status",
    ]
    cols = [c for c in preferred if c in out.columns]
    return out[cols].sort_values(["season", "game_date", "game_pk", "defense_side"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="historical_ps/postseason_all_available.parquet",
        help="pitch-level historical postseason parquet",
    )
    ap.add_argument(
        "--output",
        default="historical_ps/ps_starter_appearances.csv",
        help="one-row-per-starter CSV output",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_parquet(src)
    out = extract_starter_appearances(df)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8")

    print(
        {
            "input": str(src),
            "output": str(dst),
            "games": int(out["game_pk"].nunique()),
            "starter_appearances": int(len(out)),
            "seasons": sorted(out["season"].dropna().astype(int).unique().tolist()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

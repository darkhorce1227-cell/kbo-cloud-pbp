#!/usr/bin/env python3
"""Extract one starter appearance per team/game from historical PS pitch-level PBP.

Input is a postseason pitch-level Parquet produced by the repository's Naver PBP
pipeline (for example postseason_screen_pass.parquet).

The extractor is deliberately narrow:
- one final row is kept per plate appearance;
- defensive outs for the PA are ``post_outs - outs_when_up``;
- the pitcher on the final row of the PA receives those outs;
- the first pitcher observed for each defending team is the starter;
- starter actual IP is recorded outs / 3.

No regular-season baseline is joined here.  This file only creates the PS starter
appearance layer needed by the later historical Expected-IP-shift reconstruction.
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


def _half_to_defending_team(row: pd.Series) -> str:
    half = str(row["inning_topbot"]).strip().lower()
    if half.startswith("top") or half in {"초", "t"}:
        return str(row["home_team"])
    if half.startswith("bot") or half.startswith("bottom") or half in {"말", "b"}:
        return str(row["away_team"])
    raise ValueError(f"unrecognized inning_topbot={row['inning_topbot']!r}")


def _prepare_pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    x = df.copy()
    x["game_pk"] = x["game_pk"].astype(str)
    x["pitcher"] = x["pitcher"].astype(str)
    x["game_date"] = pd.to_datetime(x["game_date"], errors="raise")

    # Stable order within a PA.  The final pitch/event row carries the PA-close
    # state (post_outs) and the pitcher who actually completed the PA.
    x = x.sort_values(
        ["season", "game_date", "game_pk", "inning", "at_bat_number", "pitch_number"],
        kind="stable",
    )
    pa = x.groupby(["game_pk", "at_bat_number"], sort=False, as_index=False).tail(1).copy()

    pa["outs_when_up"] = pd.to_numeric(pa["outs_when_up"], errors="raise").astype(int)
    pa["post_outs"] = pd.to_numeric(pa["post_outs"], errors="raise").astype(int)
    pa["outs_recorded"] = pa["post_outs"] - pa["outs_when_up"]

    bad = pa.loc[~pa["outs_recorded"].between(0, 3)]
    if not bad.empty:
        cols = [
            "season", "game_pk", "inning", "inning_topbot", "at_bat_number",
            "pitcher", "outs_when_up", "post_outs", "outs_recorded",
        ]
        raise RuntimeError(
            "invalid PA out transition(s); sample="
            + bad[cols].head(10).to_dict(orient="records").__repr__()
        )

    pa["defending_team"] = pa.apply(_half_to_defending_team, axis=1)
    return pa


def extract_starter_appearances(df: pd.DataFrame) -> pd.DataFrame:
    pa = _prepare_pa_rows(df)

    # First observed pitcher for each defensive team/game is the starter.
    first = (
        pa.sort_values(
            ["season", "game_date", "game_pk", "inning", "at_bat_number"],
            kind="stable",
        )
        .groupby(["game_pk", "defending_team"], sort=False, as_index=False)
        .first()[
            [
                "season", "game_pk", "game_date", "home_team", "away_team",
                "defending_team", "pitcher", "pitcher_name",
            ]
        ]
        .rename(columns={"pitcher": "starter_id", "pitcher_name": "starter_name"})
    )

    # Attribute each completed PA's recorded outs to the pitcher on that PA's
    # final row.  This also handles zero-out PAs, double plays and triple plays.
    pitcher_outs = (
        pa.groupby(["game_pk", "defending_team", "pitcher"], as_index=False)["outs_recorded"]
        .sum()
        .rename(columns={"pitcher": "starter_id", "outs_recorded": "actual_outs"})
    )

    out = first.merge(
        pitcher_outs,
        on=["game_pk", "defending_team", "starter_id"],
        how="left",
        validate="one_to_one",
    )
    if out["actual_outs"].isna().any():
        raise RuntimeError("starter identified but actual outs could not be reconstructed")

    out["actual_outs"] = out["actual_outs"].astype(int)
    out["actual_ip"] = out["actual_outs"] / 3.0

    # Exactness gates: two defensive starters per complete game and no duplicate
    # starter-team key.  Incomplete historical games should fail loudly rather
    # than silently enter the shift study.
    if out.duplicated(["game_pk", "defending_team"]).any():
        raise RuntimeError("duplicate game/team starter rows")

    per_game = out.groupby("game_pk").size()
    bad_games = per_game[per_game != 2]
    if not bad_games.empty:
        raise RuntimeError(
            "expected exactly two starters per game; bad games="
            + bad_games.head(20).to_dict().__repr__()
        )

    cols = [
        "season",
        "game_date",
        "game_pk",
        "home_team",
        "away_team",
        "defending_team",
        "starter_id",
        "starter_name",
        "actual_outs",
        "actual_ip",
    ]
    return out[cols].sort_values(
        ["season", "game_date", "game_pk", "defending_team"], kind="stable"
    ).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="historical postseason pitch-level Parquet")
    ap.add_argument("--output", required=True, help="output CSV path")
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

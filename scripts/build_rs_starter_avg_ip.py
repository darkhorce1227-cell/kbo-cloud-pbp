#!/usr/bin/env python3
"""Build reproducible KBO regular-season starter average IP by player-season.

Inputs are one completed Naver starter pitching line per regular-season game-side,
collected by collect_rs_starter_lines.py. Stable Naver pcode is the authoritative
player key; names and team codes are descriptive only.

The script also audits join coverage against the validated postseason starter
artifact and reproduces the Phase-2 baseline RMSE for 2017-2025.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

YEARS = list(range(2016, 2026))
REFERENCE_2017_2025_PS_ROWS = 258
REFERENCE_2017_2025_PS_GAMES = 129
REFERENCE_BASELINE_RS_IP_RMSE = 1.84291


def _rmse(values) -> float:
    s = pd.Series(values, dtype=float)
    return float(math.sqrt((s * s).mean()))


def _norm_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _joined(values) -> str:
    vals = sorted({str(v).strip() for v in values if not pd.isna(v) and str(v).strip()})
    return "|".join(vals)


def _first_name(values) -> str:
    for value in values:
        if not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return ""


def load_rs_source(source_dir: Path, years: list[int]) -> pd.DataFrame:
    parts = []
    missing = []
    for year in years:
        path = source_dir / f"{year}_rs_starters.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        part = pd.read_csv(
            path,
            dtype={
                "game_id": str,
                "relay_game_id": str,
                "starter_pcode": str,
                "away_team_code": str,
                "home_team_code": str,
                "team_code": str,
                "opponent_code": str,
            },
        )
        parts.append(part)
    if missing:
        raise FileNotFoundError(f"missing RS source files: {missing}")
    return pd.concat(parts, ignore_index=True)


def validate_appearances(rs: pd.DataFrame, years: list[int]) -> dict:
    required = {
        "season", "game_id", "game_date", "side", "team_code",
        "starter_pcode", "starter_name", "actual_outs", "actual_ip",
    }
    missing = sorted(required - set(rs.columns))
    if missing:
        raise ValueError(f"RS source missing columns: {missing}")

    rs = rs.copy()
    rs["season"] = pd.to_numeric(rs["season"], errors="coerce")
    rs["actual_outs"] = pd.to_numeric(rs["actual_outs"], errors="coerce")
    rs["actual_ip"] = pd.to_numeric(rs["actual_ip"], errors="coerce")
    rs["starter_pcode"] = rs["starter_pcode"].map(_norm_id)

    bad_numeric = int(rs[["season", "actual_outs", "actual_ip"]].isna().any(axis=1).sum())
    empty_ids = int((rs["starter_pcode"] == "").sum())
    duplicates = int(rs.duplicated(["season", "game_id", "side"], keep=False).sum())
    bad_sides = int((~rs["side"].isin(["away", "home"])).sum())
    bad_outs = int(((rs["actual_outs"] < 0) | (rs["actual_outs"] % 1 != 0)).sum())
    bad_thirds = int(((rs["actual_ip"] * 3 - rs["actual_outs"]).abs() > 1e-9).sum())

    by_season = []
    for year in years:
        g = rs[rs["season"] == year]
        games = int(g["game_id"].nunique())
        rows = int(len(g))
        dup_game_sides = int(g.duplicated(["game_id", "side"], keep=False).sum())
        by_season.append(
            {
                "season": year,
                "games": games,
                "starter_appearances": rows,
                "two_starters_per_game": bool(rows == 2 * games and dup_game_sides == 0),
            }
        )

    failures = {
        "bad_numeric_rows": bad_numeric,
        "empty_starter_pcode": empty_ids,
        "duplicate_game_side_rows": duplicates,
        "bad_side_rows": bad_sides,
        "bad_actual_outs_rows": bad_outs,
        "bad_ip_thirds_rows": bad_thirds,
        "seasons_not_two_starters_per_game": int(
            sum(not row["two_starters_per_game"] for row in by_season)
        ),
    }
    return {
        "status": "PASS" if not any(failures.values()) else "FAIL",
        "failures": failures,
        "by_season": by_season,
    }


def build_player_season(rs: pd.DataFrame) -> pd.DataFrame:
    data = rs.copy()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["actual_outs"] = pd.to_numeric(data["actual_outs"], errors="raise").astype(int)
    data["starter_pcode"] = data["starter_pcode"].map(_norm_id)
    data["game_date"] = data["game_date"].astype(str)

    agg = (
        data.sort_values(["season", "game_date", "game_id", "side"], kind="stable")
        .groupby(["season", "starter_pcode"], as_index=False)
        .agg(
            starter_name=("starter_name", _first_name),
            name_variants=("starter_name", _joined),
            team_codes=("team_code", _joined),
            rs_starts=("game_id", "size"),
            rs_total_outs=("actual_outs", "sum"),
            rs_first_start=("game_date", "min"),
            rs_last_start=("game_date", "max"),
        )
    )
    agg["rs_starter_avg_outs"] = agg["rs_total_outs"] / agg["rs_starts"]
    agg["rs_starter_avg_ip"] = agg["rs_total_outs"] / (3.0 * agg["rs_starts"])
    return agg.sort_values(["season", "starter_pcode"], kind="stable").reset_index(drop=True)


def audit_ps_join(ps_path: Path, avg: pd.DataFrame):
    ps = pd.read_csv(ps_path, dtype={"starter_id": str, "game_pk": str})
    required = {"season", "game_pk", "starter_id", "starter_name", "actual_ip"}
    missing = sorted(required - set(ps.columns))
    if missing:
        raise ValueError(f"PS artifact missing columns: {missing}")

    ps["season"] = pd.to_numeric(ps["season"], errors="raise").astype(int)
    ps["starter_id"] = ps["starter_id"].map(_norm_id)
    ps["actual_ip"] = pd.to_numeric(ps["actual_ip"], errors="raise")

    right = avg.rename(columns={"starter_pcode": "starter_id"})
    joined = ps.merge(
        right[
            [
                "season", "starter_id", "starter_name", "name_variants", "team_codes",
                "rs_starts", "rs_total_outs", "rs_starter_avg_outs", "rs_starter_avg_ip",
            ]
        ].rename(columns={"starter_name": "rs_starter_name"}),
        on=["season", "starter_id"],
        how="left",
        validate="many_to_one",
    )
    joined["rs_link_ok"] = joined["rs_starter_avg_ip"].notna()
    joined["baseline_rs_ip_error"] = joined["rs_starter_avg_ip"] - joined["actual_ip"]

    eval_ = joined[joined["season"].between(2017, 2025)].copy()
    baseline_rmse = _rmse(eval_["baseline_rs_ip_error"]) if len(eval_) else None
    return joined, {
        "ps_rows_2016_2025": int(len(joined[joined["season"].between(2016, 2025)])),
        "ps_linked_rows_2016_2025": int(
            joined.loc[joined["season"].between(2016, 2025), "rs_link_ok"].sum()
        ),
        "ps_missing_rs_links_2016_2025": int(
            (~joined.loc[joined["season"].between(2016, 2025), "rs_link_ok"]).sum()
        ),
        "ps_games_2017_2025": int(eval_["game_pk"].nunique()),
        "ps_rows_2017_2025": int(len(eval_)),
        "ps_linked_rows_2017_2025": int(eval_["rs_link_ok"].sum()),
        "baseline_rs_ip_rmse_2017_2025": baseline_rmse,
        "reference_baseline_rs_ip_rmse": REFERENCE_BASELINE_RS_IP_RMSE,
        "reference_rmse_delta": (
            baseline_rmse - REFERENCE_BASELINE_RS_IP_RMSE
            if baseline_rmse is not None
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="historical_rs/source")
    ap.add_argument("--ps-input", default="historical_ps/ps_starter_appearances.csv")
    ap.add_argument("--output-dir", default="historical_rs")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    source_dir = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rs = load_rs_source(source_dir, years)
    validation = validate_appearances(rs, years)
    if validation["status"] != "PASS":
        raise ValueError(f"RS appearance validation failed: {validation['failures']}")

    rs["starter_pcode"] = rs["starter_pcode"].map(_norm_id)
    avg = build_player_season(rs)
    linked, join = audit_ps_join(Path(args.ps_input), avg)

    expected_games = 720 * len(years) if years == YEARS else None
    aggregate_checks = {
        "all_source_seasons_present": sorted(rs["season"].astype(int).unique().tolist()) == years,
        "all_rows_two_per_game": int(len(rs)) == 2 * int(rs["game_id"].nunique()),
        "expected_720_games_per_season_2016_2025": (
            int(rs["game_id"].nunique()) == expected_games if expected_games is not None else None
        ),
        "ps_2017_2025_expected_129_games": join["ps_games_2017_2025"] == REFERENCE_2017_2025_PS_GAMES,
        "ps_2017_2025_expected_258_rows": join["ps_rows_2017_2025"] == REFERENCE_2017_2025_PS_ROWS,
        "ps_2017_2025_all_linked_to_rs": join["ps_linked_rows_2017_2025"] == REFERENCE_2017_2025_PS_ROWS,
        "ps_2016_2025_all_linked_to_rs": join["ps_missing_rs_links_2016_2025"] == 0,
        "baseline_rmse_matches_phase2_within_0_001": (
            abs(join["reference_rmse_delta"]) <= 0.001
            if join["reference_rmse_delta"] is not None
            else False
        ),
    }
    required_checks = [v for v in aggregate_checks.values() if v is not None]
    status = "PASS" if all(required_checks) else "FAIL"

    rs_out = out / "rs_starter_appearances_2016_2025.csv"
    avg_out = out / "rs_starter_avg_ip_by_player_season.csv"
    link_out = out / "ps_rs_starter_avg_ip_join_audit_2016_2025.csv"
    validation_out = out / "rs_starter_avg_ip_validation.json"

    rs.sort_values(["season", "game_date", "game_id", "side"], kind="stable").to_csv(
        rs_out, index=False
    )
    avg.to_csv(avg_out, index=False)
    linked.sort_values(["season", "game_date", "game_pk", "home_away"], kind="stable").to_csv(
        link_out, index=False
    )

    report = {
        "status": status,
        "definition": "same-season player regular-season starter average IP",
        "player_key": "Naver stable pcode / postseason starter_id",
        "source": "Naver inning-1 relay completed Lineup.pitcher; seqno=1 starter",
        "source_years": years,
        "rs_games": int(rs["game_id"].nunique()),
        "rs_starter_appearances": int(len(rs)),
        "player_season_rows": int(len(avg)),
        "appearance_validation": validation,
        "ps_join_audit": join,
        "checks": aggregate_checks,
        "outputs": {
            "rs_appearances": str(rs_out),
            "player_season_avg": str(avg_out),
            "ps_join_audit": str(link_out),
        },
    }
    validation_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

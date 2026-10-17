#!/usr/bin/env python3
"""Build reproducible KBO regular-season starter average IP by player-season.

Inputs are one completed Naver starter pitching line per regular-season game-side,
collected by collect_rs_starter_lines.py. Stable Naver pcode is the authoritative
player key; names/team codes are descriptive only.

A postseason starter can legitimately have zero same-season regular-season starts.
Those rows must remain explicit NO_SAME_SEASON_RS_START_HISTORY rather than being
filled by name, team average, relief IP, or any other inferred fallback.
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
HISTORICAL_REFERENCE_BASELINE_RS_IP_RMSE = 1.84291

# Reverified against current Naver inning-1 postseason relay using the same
# seqno=1 starter rule as the saved Phase-2 collector.
EXPECTED_NO_RS_START_HISTORY = {
    (2018, "33331023HHWO02018", "68703", "박주홍"),
    (2022, "77771105SKWO02022", "67603", "이승호"),
}


def _rmse(values) -> float:
    s = pd.Series(values, dtype=float).dropna()
    return float(math.sqrt((s * s).mean())) if len(s) else float("nan")


def _mae(values) -> float:
    s = pd.Series(values, dtype=float).dropna()
    return float(s.abs().mean()) if len(s) else float("nan")


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
        parts.append(
            pd.read_csv(
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
        )
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

    x = rs.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["actual_outs"] = pd.to_numeric(x["actual_outs"], errors="coerce")
    x["actual_ip"] = pd.to_numeric(x["actual_ip"], errors="coerce")
    x["starter_pcode"] = x["starter_pcode"].map(_norm_id)

    failures = {
        "bad_numeric_rows": int(x[["season", "actual_outs", "actual_ip"]].isna().any(axis=1).sum()),
        "empty_starter_pcode": int((x["starter_pcode"] == "").sum()),
        "duplicate_game_side_rows": int(x.duplicated(["season", "game_id", "side"], keep=False).sum()),
        "bad_side_rows": int((~x["side"].isin(["away", "home"])).sum()),
        "bad_actual_outs_rows": int(((x["actual_outs"] < 0) | (x["actual_outs"] % 1 != 0)).sum()),
        "bad_ip_thirds_rows": int(((x["actual_ip"] * 3 - x["actual_outs"]).abs() > 1e-9).sum()),
    }

    by_season = []
    for year in years:
        g = x[x["season"] == year]
        games = int(g["game_id"].nunique())
        rows = int(len(g))
        two_per_game = bool(
            rows == 2 * games
            and int(g.duplicated(["game_id", "side"], keep=False).sum()) == 0
        )
        by_season.append(
            {
                "season": year,
                "games": games,
                "starter_appearances": rows,
                "two_starters_per_game": two_per_game,
            }
        )
    failures["seasons_not_two_starters_per_game"] = int(
        sum(not row["two_starters_per_game"] for row in by_season)
    )
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
    required = {
        "season", "game_date", "game_pk", "team", "home_away",
        "starter_id", "starter_name", "actual_ip",
    }
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
    joined["rs_link_status"] = joined["rs_starter_avg_ip"].notna().map(
        {True: "MATCHED", False: "NO_SAME_SEASON_RS_START_HISTORY"}
    )
    joined["baseline_rs_ip_error"] = joined["rs_starter_avg_ip"] - joined["actual_ip"]

    all_16_25 = joined[joined["season"].between(2016, 2025)].copy()
    eval_ = joined[joined["season"].between(2017, 2025)].copy()
    matched_eval = eval_[eval_["rs_link_status"] == "MATCHED"].copy()
    missing_rows = eval_[eval_["rs_link_status"] != "MATCHED"].copy()

    observed_missing = {
        (int(r.season), str(r.game_pk), str(r.starter_id), str(r.starter_name))
        for r in missing_rows.itertuples(index=False)
    }
    missing_details = [
        {
            "season": int(r.season),
            "game_date": str(r.game_date),
            "game_pk": str(r.game_pk),
            "team": str(r.team),
            "starter_id": str(r.starter_id),
            "starter_name": str(r.starter_name),
            "ps_actual_ip": float(r.actual_ip),
            "status": "NO_SAME_SEASON_RS_START_HISTORY",
        }
        for r in missing_rows.itertuples(index=False)
    ]

    baseline_rmse = _rmse(matched_eval["baseline_rs_ip_error"])
    baseline_mae = _mae(matched_eval["baseline_rs_ip_error"])
    return joined, {
        "ps_rows_2016_2025": int(len(all_16_25)),
        "ps_linked_rows_2016_2025": int((all_16_25["rs_link_status"] == "MATCHED").sum()),
        "ps_no_rs_start_history_2016_2025": int((all_16_25["rs_link_status"] != "MATCHED").sum()),
        "ps_games_2017_2025": int(eval_["game_pk"].nunique()),
        "ps_rows_2017_2025": int(len(eval_)),
        "ps_linked_rows_2017_2025": int(len(matched_eval)),
        "ps_no_rs_start_history_2017_2025": int(len(missing_rows)),
        "no_rs_start_history_details": missing_details,
        "expected_no_rs_start_history_exact_match": observed_missing == EXPECTED_NO_RS_START_HISTORY,
        "baseline_rs_ip_rmse_matched_2017_2025": baseline_rmse,
        "baseline_rs_ip_mae_matched_2017_2025": baseline_mae,
        "baseline_sample_n": int(len(matched_eval)),
        "historical_reference_baseline_rs_ip_rmse": HISTORICAL_REFERENCE_BASELINE_RS_IP_RMSE,
        "historical_reference_rmse_delta_on_current_matched_sample": (
            baseline_rmse - HISTORICAL_REFERENCE_BASELINE_RS_IP_RMSE
        ),
        "historical_reference_status": "NON_REPRODUCIBLE_AS_258_ROW_RS_STARTER_BASELINE",
        "historical_reference_reason": (
            "Saved Phase-2 builder requires every PS starter to have same-season RS starter history, "
            "but raw Naver reconstruction has two verified PS starters with zero same-season RS starts. "
            "No inference/fallback is applied here."
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
    checks = {
        "all_source_seasons_present": sorted(rs["season"].astype(int).unique().tolist()) == years,
        "all_rows_two_per_game": int(len(rs)) == 2 * int(rs["game_id"].nunique()),
        "expected_720_games_per_season_2016_2025": (
            int(rs["game_id"].nunique()) == expected_games if expected_games is not None else None
        ),
        "ps_2017_2025_expected_129_games": join["ps_games_2017_2025"] == REFERENCE_2017_2025_PS_GAMES,
        "ps_2017_2025_expected_258_rows": join["ps_rows_2017_2025"] == REFERENCE_2017_2025_PS_ROWS,
        "ps_2017_2025_exactly_256_rs_history_matches": join["ps_linked_rows_2017_2025"] == 256,
        "ps_2017_2025_exactly_2_no_rs_start_history": join["ps_no_rs_start_history_2017_2025"] == 2,
        "no_rs_start_history_rows_match_reverified_naver_cases": join["expected_no_rs_start_history_exact_match"],
    }
    required_checks = [v for v in checks.values() if v is not None]
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
        "checks": checks,
        "policy": {
            "zero_rs_start_history": "UNAVAILABLE; no inferred fallback in Step 2",
            "historical_phase2_reference": "retained for audit only; not a production parity target",
        },
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

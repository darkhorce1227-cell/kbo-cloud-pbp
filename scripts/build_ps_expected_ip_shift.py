#!/usr/bin/env python3
"""Rebuild the postseason Expected-IP simple-shift Challenger from audited inputs.

Primary matched-sample formula for target season Y:

    candidate_ps_eip
      = same-season player RS starter-average IP
      + mean_{PS seasons < Y}(actual PS IP - same-season player RS starter-average IP)

Only postseason starter appearances with genuine same-season RS start history enter
that formula. A PS starter with zero same-season RS starts is never silently filled.
Those cases are emitted separately as cold-start sensitivity rows using team/league
RS starter means; they are descriptive only and cannot promote the model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = {
    "sequential_seasons": list(range(2017, 2026)),
    "games": 129,
    "starter_appearances": 258,
    "baseline_rs_ip_rmse": 1.84291,
    "candidate_rmse": 1.67696,
    "candidate_mae": 1.32339,
    "candidate_rmse_season_wins": 9,
    "mse_improvement_ci": [-0.859, -0.312],
}


def rmse(errors) -> float:
    s = pd.Series(errors, dtype=float).dropna()
    return float(math.sqrt(float((s * s).mean()))) if len(s) else float("nan")


def mae(errors) -> float:
    s = pd.Series(errors, dtype=float).dropna()
    return float(s.abs().mean()) if len(s) else float("nan")


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def bootstrap_game_mse_delta(preds: pd.DataFrame, reps: int, seed: int) -> dict:
    """Game-cluster bootstrap of MSE(candidate) - MSE(baseline)."""
    by_game = []
    for game_pk, g in preds.groupby("game_pk", sort=False):
        base_sq = np.square(g["baseline_error"].to_numpy(dtype=float))
        cand_sq = np.square(g["candidate_error"].to_numpy(dtype=float))
        by_game.append((str(game_pk), float(base_sq.sum()), float(cand_sq.sum()), int(len(g))))

    if not by_game:
        return {"reps": reps, "seed": seed, "point": None, "ci_low": None, "ci_high": None}

    base_sum = np.array([x[1] for x in by_game], dtype=float)
    cand_sum = np.array([x[2] for x in by_game], dtype=float)
    counts = np.array([x[3] for x in by_game], dtype=float)
    n_games = len(by_game)
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n_games, size=n_games)
        denom = counts[idx].sum()
        deltas[i] = cand_sum[idx].sum() / denom - base_sum[idx].sum() / denom

    point = float(np.square(preds["candidate_error"]).mean() - np.square(preds["baseline_error"]).mean())
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "cluster": "game_pk",
        "reps": int(reps),
        "seed": int(seed),
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-input", default="historical_ps/ps_starter_appearances.csv")
    ap.add_argument("--rs-avg-input", default="historical_rs/rs_starter_avg_ip_by_player_season.csv")
    ap.add_argument("--rs-app-input", default="historical_rs/rs_starter_appearances_2016_2025.csv")
    ap.add_argument("--output-dir", default="research/postseason")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--bootstrap-reps", type=int, default=50000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261017)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ps = pd.read_csv(args.ps_input, dtype={"starter_id": str, "game_pk": str})
    rs_avg = pd.read_csv(args.rs_avg_input, dtype={"starter_pcode": str})
    rs_app = pd.read_csv(args.rs_app_input, dtype={"starter_pcode": str, "game_id": str})

    ps["season"] = pd.to_numeric(ps["season"], errors="raise").astype(int)
    ps["starter_id"] = ps["starter_id"].map(normalize_id)
    ps["actual_ip"] = pd.to_numeric(ps["actual_ip"], errors="raise")
    rs_avg["season"] = pd.to_numeric(rs_avg["season"], errors="raise").astype(int)
    rs_avg["starter_pcode"] = rs_avg["starter_pcode"].map(normalize_id)
    rs_app["season"] = pd.to_numeric(rs_app["season"], errors="raise").astype(int)
    rs_app["actual_outs"] = pd.to_numeric(rs_app["actual_outs"], errors="raise").astype(int)

    years = list(range(args.start_year, args.end_year + 1))
    ps = ps[ps["season"].isin(years)].copy()
    rs_avg = rs_avg[rs_avg["season"].isin(years)].copy()
    rs_app = rs_app[rs_app["season"].isin(years)].copy()

    right = rs_avg.rename(columns={"starter_pcode": "starter_id", "starter_name": "rs_starter_name"})
    linked = ps.merge(
        right[[
            "season", "starter_id", "rs_starter_name", "name_variants", "team_codes",
            "rs_starts", "rs_total_outs", "rs_starter_avg_outs", "rs_starter_avg_ip",
        ]],
        on=["season", "starter_id"], how="left", validate="many_to_one",
    )
    linked["rs_link_status"] = linked["rs_starter_avg_ip"].notna().map(
        {True: "MATCHED", False: "NO_SAME_SEASON_RS_START_HISTORY"}
    )
    linked["ps_minus_rs_avg_ip"] = linked["actual_ip"] - linked["rs_starter_avg_ip"]

    matched = linked[linked["rs_link_status"] == "MATCHED"].copy()
    cold = linked[linked["rs_link_status"] != "MATCHED"].copy()

    # Leakage-free expanding shift. Target Y sees only matched PS appearances from seasons < Y.
    shift_rows = []
    pred_parts = []
    for target in range(args.start_year + 1, args.end_year + 2):
        hist = matched[matched["season"] < target].copy()
        if hist.empty:
            continue
        shift = float(hist["ps_minus_rs_avg_ip"].mean())
        shift_rows.append({
            "target_season": int(target),
            "source_start_season": int(hist["season"].min()),
            "source_through_season": int(hist["season"].max()),
            "historical_matched_ps_starter_appearances": int(len(hist)),
            "historical_ps_games_with_matched_starters": int(hist["game_pk"].nunique()),
            "historical_ps_mean_ip_shift": shift,
        })

        current = matched[matched["season"] == target].copy()
        if current.empty:
            continue
        current["historical_ps_mean_ip_shift"] = shift
        current["baseline_rs_eip"] = current["rs_starter_avg_ip"]
        current["candidate_ps_eip"] = current["rs_starter_avg_ip"] + shift
        current["baseline_error"] = current["baseline_rs_eip"] - current["actual_ip"]
        current["candidate_error"] = current["candidate_ps_eip"] - current["actual_ip"]
        pred_parts.append(current)

    shifts = pd.DataFrame(shift_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    per_season_rows = []
    for season, g in preds.groupby("season", sort=True):
        base = rmse(g["baseline_error"])
        cand = rmse(g["candidate_error"])
        per_season_rows.append({
            "season": int(season),
            "games": int(g["game_pk"].nunique()),
            "matched_starter_appearances": int(len(g)),
            "historical_shift": float(g["historical_ps_mean_ip_shift"].iloc[0]),
            "baseline_rs_ip_rmse": base,
            "baseline_rs_ip_mae": mae(g["baseline_error"]),
            "candidate_rmse": cand,
            "candidate_mae": mae(g["candidate_error"]),
            "candidate_mse_delta_vs_baseline": float(np.square(g["candidate_error"]).mean() - np.square(g["baseline_error"]).mean()),
            "candidate_beats_baseline_rmse": bool(cand < base),
        })
    per_season = pd.DataFrame(per_season_rows)

    # Cold-start sensitivity. Same target-season historical shift, but no player RS baseline exists.
    team_means = (
        rs_app.groupby(["season", "team_code"], as_index=False)
        .agg(team_rs_starts=("game_id", "size"), team_rs_total_outs=("actual_outs", "sum"))
    )
    team_means["team_rs_starter_avg_ip"] = team_means["team_rs_total_outs"] / (3.0 * team_means["team_rs_starts"])
    league_means = (
        rs_app.groupby("season", as_index=False)
        .agg(league_rs_starts=("game_id", "size"), league_rs_total_outs=("actual_outs", "sum"))
    )
    league_means["league_rs_starter_avg_ip"] = league_means["league_rs_total_outs"] / (3.0 * league_means["league_rs_starts"])

    cold_rows = []
    shift_map = shifts.set_index("target_season")["historical_ps_mean_ip_shift"].to_dict()
    for r in cold.itertuples(index=False):
        target = int(r.season)
        tm = team_means[(team_means["season"] == target) & (team_means["team_code"].astype(str) == str(r.team))]
        lm = league_means[league_means["season"] == target]
        shift = shift_map.get(target)
        team_base = float(tm.iloc[0]["team_rs_starter_avg_ip"]) if len(tm) else None
        league_base = float(lm.iloc[0]["league_rs_starter_avg_ip"]) if len(lm) else None
        team_candidate = team_base + shift if team_base is not None and shift is not None else None
        league_candidate = league_base + shift if league_base is not None and shift is not None else None
        cold_rows.append({
            "season": target,
            "game_date": r.game_date,
            "game_pk": r.game_pk,
            "team": r.team,
            "opponent": r.opponent,
            "starter_id": r.starter_id,
            "starter_name": r.starter_name,
            "actual_ip": float(r.actual_ip),
            "historical_ps_mean_ip_shift": shift,
            "team_rs_starter_avg_ip": team_base,
            "team_shift_candidate_eip": team_candidate,
            "team_shift_error": (team_candidate - float(r.actual_ip)) if team_candidate is not None else None,
            "league_rs_starter_avg_ip": league_base,
            "league_shift_candidate_eip": league_candidate,
            "league_shift_error": (league_candidate - float(r.actual_ip)) if league_candidate is not None else None,
            "status": "DESCRIPTIVE_COLD_START_ONLY",
        })
    cold_df = pd.DataFrame(cold_rows)

    bootstrap = bootstrap_game_mse_delta(preds, args.bootstrap_reps, args.bootstrap_seed)
    shift_2026 = None
    if not shifts.empty and 2026 in set(shifts["target_season"].astype(int)):
        shift_2026 = float(shifts.loc[shifts["target_season"] == 2026, "historical_ps_mean_ip_shift"].iloc[0])

    metrics = {
        "status": "PASS",
        "version": "ps-expected-ip-simple-shift-audited-v0.2",
        "formula": "player same-season RS starter avg IP + expanding mean(previous matched PS actual IP - same-season player RS starter avg IP)",
        "leakage_rule": "target season Y uses only matched PS starter appearances from seasons < Y",
        "source_years": years,
        "ps_starter_appearances_2016_2025": int(len(linked)),
        "matched_ps_starter_appearances_2016_2025": int(len(matched)),
        "cold_start_ps_starter_appearances_2016_2025": int(len(cold)),
        "sequential_eval_seasons": sorted(preds["season"].astype(int).unique().tolist()),
        "sequential_games": int(preds["game_pk"].nunique()),
        "sequential_matched_starter_appearances": int(len(preds)),
        "baseline_rs_ip_rmse": rmse(preds["baseline_error"]),
        "baseline_rs_ip_mae": mae(preds["baseline_error"]),
        "candidate_rmse": rmse(preds["candidate_error"]),
        "candidate_mae": mae(preds["candidate_error"]),
        "candidate_mse_delta_vs_baseline": float(np.square(preds["candidate_error"]).mean() - np.square(preds["baseline_error"]).mean()),
        "candidate_rmse_season_wins": int(per_season["candidate_beats_baseline_rmse"].sum()),
        "candidate_rmse_seasons_evaluated": int(len(per_season)),
        "bootstrap_game_mse_delta": bootstrap,
        "shift_for_2026": shift_2026,
        "cold_start": {
            "n": int(len(cold_df)),
            "policy": "not part of matched-sample Challenger; no automatic production fallback",
            "team_shift_rmse": rmse(cold_df["team_shift_error"]) if len(cold_df) else None,
            "league_shift_rmse": rmse(cold_df["league_shift_error"]) if len(cold_df) else None,
            "interpretation": "n=2 only; sensitivity/descriptive, not validation",
        },
        "historical_reference_phase2": REFERENCE,
        "historical_reference_status": "AUDIT_ONLY_NON_REPRODUCIBLE_AS_258_ROW_PLAYER_RS_BASELINE",
    }

    checks = {
        "ps_total_286": len(linked) == 286,
        "matched_284": len(matched) == 284,
        "cold_start_2": len(cold) == 2,
        "sequential_eval_2017_2025": metrics["sequential_eval_seasons"] == list(range(2017, 2026)),
        "sequential_games_129": metrics["sequential_games"] == 129,
        "sequential_matched_rows_256": metrics["sequential_matched_starter_appearances"] == 256,
        "nine_seasons_evaluated": metrics["candidate_rmse_seasons_evaluated"] == 9,
        "shift_2026_available": shift_2026 is not None,
        "all_prediction_inputs_nonmissing": not preds[["rs_starter_avg_ip", "historical_ps_mean_ip_shift", "actual_ip"]].isna().any().any(),
    }
    metrics["checks"] = checks
    if not all(checks.values()):
        metrics["status"] = "FAIL"

    # Research artifacts.
    linked.sort_values(["season", "game_date", "game_pk", "home_away"], kind="stable").to_csv(
        out / "ps_rs_starter_avg_ip.csv", index=False
    )
    shifts.to_csv(out / "ps_expected_ip_shift_by_target_season.csv", index=False)
    preds.sort_values(["season", "game_date", "game_pk", "home_away"], kind="stable").to_csv(
        out / "ps_expected_ip_simple_shift_predictions_2017_2025.csv", index=False
    )
    per_season.to_csv(out / "ps_expected_ip_simple_shift_by_season.csv", index=False)
    cold_df.to_csv(out / "ps_expected_ip_cold_start_sensitivity.csv", index=False)
    (out / "ps_expected_ip_simple_shift_validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if metrics["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

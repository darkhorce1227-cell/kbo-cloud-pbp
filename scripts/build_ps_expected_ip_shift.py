#!/usr/bin/env python3
"""Build the reproducible historical postseason Expected-IP simple-shift artifact.

Formula for target season Y:
  candidate_eip = player RS starter-average IP in Y
                  + mean(previous PS actual IP - same-season player RS starter-average IP)
where `previous PS` contains only seasons < Y.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

REFERENCE = {
    "sequential_seasons": list(range(2017, 2026)),
    "games": 129,
    "starter_appearances": 258,
    "baseline_rs_ip_rmse": 1.84291,
    "candidate_rmse": 1.67696,
    "candidate_mae": 1.32339,
    "candidate_rmse_season_wins": 9,
}


def rmse(values):
    s = pd.Series(values, dtype=float)
    return float(math.sqrt((s * s).mean()))


def load_years(source_dir: Path, years):
    rs_parts, ps_parts = [], []
    missing = []
    for year in years:
        rp = source_dir / f"{year}_rs_starters.csv"
        pp = source_dir / f"{year}_ps_starters.csv"
        if not rp.exists() or not pp.exists():
            missing.append(year)
            continue
        rs_parts.append(pd.read_csv(rp, dtype={"starter_pcode": str, "game_id": str, "relay_game_id": str}))
        ps_parts.append(pd.read_csv(pp, dtype={"starter_pcode": str, "game_id": str, "relay_game_id": str}))
    if missing:
        raise FileNotFoundError(f"missing source years: {missing}")
    return pd.concat(rs_parts, ignore_index=True), pd.concat(ps_parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="research/postseason-eip/source")
    ap.add_argument("--output-dir", default="research/postseason-eip")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    source_dir = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rs, ps = load_years(source_dir, years)
    for label, frame in (("RS", rs), ("PS", ps)):
        dup = frame.duplicated(["season", "game_id", "side"], keep=False)
        if dup.any():
            raise ValueError(f"{label} duplicate game-side rows: {int(dup.sum())}")
        if frame["actual_ip"].isna().any():
            raise ValueError(f"{label} has missing actual_ip")
        if (frame["starter_pcode"].astype(str).str.len() == 0).any():
            raise ValueError(f"{label} has missing starter_pcode")

    rs_avg = (
        rs.groupby(["season", "starter_pcode", "starter_name"], as_index=False)
        .agg(rs_starter_avg_ip=("actual_ip", "mean"), rs_starts=("actual_ip", "size"))
    )
    linked = ps.merge(
        rs_avg[["season", "starter_pcode", "rs_starter_avg_ip", "rs_starts"]],
        on=["season", "starter_pcode"],
        how="left",
        validate="many_to_one",
    )
    linked["link_ok"] = linked["rs_starter_avg_ip"].notna()
    linked["ps_minus_rs_avg_ip"] = linked["actual_ip"] - linked["rs_starter_avg_ip"]

    missing_link = linked.loc[~linked["link_ok"], ["season", "game_id", "side", "starter_pcode", "starter_name"]]
    if not missing_link.empty:
        missing_link.to_csv(out / "ps_eip_missing_rs_links.csv", index=False)
        raise ValueError(f"PS starter appearances without same-season RS starter history: {len(missing_link)}")

    shift_rows = []
    pred_parts = []
    min_year = min(years)
    for target in range(min_year + 1, max(years) + 2):
        history = linked[linked["season"] < target]
        if history.empty:
            continue
        shift = float(history["ps_minus_rs_avg_ip"].mean())
        shift_rows.append({
            "target_season": target,
            "source_start_season": int(history["season"].min()),
            "source_through_season": int(history["season"].max()),
            "historical_ps_starter_appearances": int(len(history)),
            "historical_ps_games": int(history["game_id"].nunique()),
            "historical_ps_mean_ip_shift": shift,
        })
        current = linked[linked["season"] == target].copy()
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

    per_season = []
    for season, g in preds.groupby("season"):
        base_rmse = rmse(g["baseline_error"])
        cand_rmse = rmse(g["candidate_error"])
        per_season.append({
            "season": int(season),
            "games": int(g["game_id"].nunique()),
            "starter_appearances": int(len(g)),
            "historical_shift": float(g["historical_ps_mean_ip_shift"].iloc[0]),
            "baseline_rs_ip_rmse": base_rmse,
            "candidate_rmse": cand_rmse,
            "candidate_mae": float(g["candidate_error"].abs().mean()),
            "candidate_beats_baseline_rmse": bool(cand_rmse < base_rmse),
        })
    season_df = pd.DataFrame(per_season)

    metrics = {
        "version": "ps-expected-ip-simple-shift-recovery-v0.1",
        "formula": "RS starter avg IP + mean(previous-season PS actual IP - same-season player RS starter avg IP)",
        "source_years": years,
        "linked_ps_games_2016_2025": int(linked["game_id"].nunique()),
        "linked_ps_starter_appearances_2016_2025": int(len(linked)),
        "sequential_eval_seasons": sorted(preds["season"].astype(int).unique().tolist()),
        "sequential_games": int(preds["game_id"].nunique()),
        "sequential_starter_appearances": int(len(preds)),
        "baseline_rs_ip_rmse": rmse(preds["baseline_error"]),
        "baseline_rs_ip_mae": float(preds["baseline_error"].abs().mean()),
        "candidate_rmse": rmse(preds["candidate_error"]),
        "candidate_mae": float(preds["candidate_error"].abs().mean()),
        "candidate_rmse_season_wins": int(season_df["candidate_beats_baseline_rmse"].sum()),
        "candidate_rmse_seasons_evaluated": int(len(season_df)),
        "shift_for_2026": float(shifts.loc[shifts["target_season"] == 2026, "historical_ps_mean_ip_shift"].iloc[0]) if 2026 in set(shifts["target_season"]) else None,
        "reference_phase2": REFERENCE,
    }
    metrics["reference_delta"] = {
        "games": metrics["sequential_games"] - REFERENCE["games"],
        "starter_appearances": metrics["sequential_starter_appearances"] - REFERENCE["starter_appearances"],
        "baseline_rs_ip_rmse": metrics["baseline_rs_ip_rmse"] - REFERENCE["baseline_rs_ip_rmse"],
        "candidate_rmse": metrics["candidate_rmse"] - REFERENCE["candidate_rmse"],
        "candidate_mae": metrics["candidate_mae"] - REFERENCE["candidate_mae"],
        "candidate_rmse_season_wins": metrics["candidate_rmse_season_wins"] - REFERENCE["candidate_rmse_season_wins"],
    }

    rs.sort_values(["season", "game_date", "game_id", "side"]).to_csv(out / "rs_starter_appearances_2016_2025.csv", index=False)
    ps.sort_values(["season", "game_date", "game_id", "side"]).to_csv(out / "ps_starter_appearances_2016_2025.csv", index=False)
    linked.sort_values(["season", "game_date", "game_id", "side"]).to_csv(out / "ps_starter_linked_2016_2025.csv", index=False)
    shifts.to_csv(out / "ps_expected_ip_shift_by_target_season.csv", index=False)
    preds.sort_values(["season", "game_date", "game_id", "side"]).to_csv(out / "ps_expected_ip_sequential_predictions_2017_2025.csv", index=False)
    season_df.to_csv(out / "ps_expected_ip_validation_by_season.csv", index=False)
    (out / "ps_expected_ip_validation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

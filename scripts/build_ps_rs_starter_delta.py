#!/usr/bin/env python3
"""Join validated postseason starter appearances to same-season RS starter-average IP.

Step 3 only. This script does NOT build any expanding postseason shift or forecast.

Authoritative join key:
  (season, Naver player id)

Output delta:
  delta_ps_minus_rs_ip = actual_PS_IP - same-season_RS_starter_avg_IP

If a postseason starter has no same-season regular-season starts, the baseline and
Delta remain unavailable. No name match, team average, relief-IP, or other fallback
is allowed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

EXPECTED_MISSING_2017_2025 = {
    (2018, "33331023HHWO02018", "68703", "박주홍"),
    (2022, "77771105SKWO02022", "67603", "이승호"),
}


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, tol=1e-10) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-input", default="historical_ps/ps_starter_appearances.csv")
    ap.add_argument("--ps-validation", default="historical_ps/ps_starter_appearances_validation.json")
    ap.add_argument("--rs-input", default="historical_rs/rs_starter_avg_ip_by_player_season.csv")
    ap.add_argument("--rs-validation", default="historical_rs/rs_starter_avg_ip_validation.json")
    ap.add_argument("--output-dir", default="research/postseason_expected_ip")
    args = ap.parse_args()

    ps_path = Path(args.ps_input)
    rs_path = Path(args.rs_input)
    ps_validation_path = Path(args.ps_validation)
    rs_validation_path = Path(args.rs_validation)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ps_validation = load_json(ps_validation_path)
    rs_validation = load_json(rs_validation_path)
    if ps_validation.get("status") != "PASS":
        raise ValueError("upstream PS starter artifact is not PASS")
    if rs_validation.get("status") != "PASS":
        raise ValueError("upstream RS starter-average artifact is not PASS")

    ps = pd.read_csv(ps_path, dtype={"game_pk": str, "starter_id": str})
    rs = pd.read_csv(rs_path, dtype={"starter_pcode": str})

    ps_required = {
        "season", "game_date", "game_pk", "team", "opponent", "home_away",
        "starter_id", "starter_name", "actual_outs", "actual_ip",
    }
    rs_required = {
        "season", "starter_pcode", "starter_name", "name_variants", "team_codes",
        "rs_starts", "rs_total_outs", "rs_starter_avg_outs", "rs_starter_avg_ip",
    }
    if ps_required - set(ps.columns):
        raise ValueError(f"PS missing columns: {sorted(ps_required - set(ps.columns))}")
    if rs_required - set(rs.columns):
        raise ValueError(f"RS missing columns: {sorted(rs_required - set(rs.columns))}")

    ps = ps.copy()
    rs = rs.copy()
    ps["season"] = pd.to_numeric(ps["season"], errors="raise").astype(int)
    rs["season"] = pd.to_numeric(rs["season"], errors="raise").astype(int)
    ps["starter_id"] = ps["starter_id"].map(norm_id)
    rs["starter_pcode"] = rs["starter_pcode"].map(norm_id)
    ps["actual_outs"] = pd.to_numeric(ps["actual_outs"], errors="raise").astype(int)
    ps["actual_ip"] = pd.to_numeric(ps["actual_ip"], errors="raise")
    rs["rs_starts"] = pd.to_numeric(rs["rs_starts"], errors="raise").astype(int)
    rs["rs_total_outs"] = pd.to_numeric(rs["rs_total_outs"], errors="raise").astype(int)
    rs["rs_starter_avg_outs"] = pd.to_numeric(rs["rs_starter_avg_outs"], errors="raise")
    rs["rs_starter_avg_ip"] = pd.to_numeric(rs["rs_starter_avg_ip"], errors="raise")

    duplicate_rs_keys = int(rs.duplicated(["season", "starter_pcode"], keep=False).sum())
    if duplicate_rs_keys:
        raise ValueError(f"duplicate RS player-season keys: {duplicate_rs_keys}")

    right = rs.rename(columns={
        "starter_pcode": "starter_id",
        "starter_name": "rs_starter_name",
    })
    joined = ps.merge(
        right[
            [
                "season", "starter_id", "rs_starter_name", "name_variants", "team_codes",
                "rs_starts", "rs_total_outs", "rs_starter_avg_outs", "rs_starter_avg_ip",
            ]
        ],
        on=["season", "starter_id"],
        how="left",
        validate="many_to_one",
    )

    joined["rs_link_status"] = joined["rs_starter_avg_ip"].notna().map({
        True: "MATCHED",
        False: "NO_SAME_SEASON_RS_START_HISTORY",
    })
    joined["delta_ps_minus_rs_ip"] = joined["actual_ip"] - joined["rs_starter_avg_ip"]

    # Preserve upstream provenance while keeping the core research fields first.
    preferred = [
        "season", "game_date", "game_pk", "team", "opponent", "home_away",
        "starter_id", "starter_name", "actual_outs", "actual_ip",
        "rs_link_status", "rs_starts", "rs_total_outs", "rs_starter_avg_outs",
        "rs_starter_avg_ip", "delta_ps_minus_rs_ip", "rs_starter_name",
        "name_variants", "team_codes",
    ]
    remaining = [c for c in joined.columns if c not in preferred]
    joined = joined[preferred + remaining]

    matched = joined[joined["rs_link_status"] == "MATCHED"].copy()
    unavailable = joined[joined["rs_link_status"] != "MATCHED"].copy()
    eval_all = joined[joined["season"].between(2017, 2025)].copy()
    eval_matched = eval_all[eval_all["rs_link_status"] == "MATCHED"].copy()
    eval_unavailable = eval_all[eval_all["rs_link_status"] != "MATCHED"].copy()

    observed_missing = {
        (int(r.season), str(r.game_pk), str(r.starter_id), str(r.starter_name))
        for r in eval_unavailable.itertuples(index=False)
    }

    # Row-level arithmetic validations.
    ps_ip_bad = int(((matched["actual_ip"] * 3.0 - matched["actual_outs"]).abs() > 1e-9).sum())
    rs_avg_formula_bad = int((
        matched.apply(
            lambda r: not close(
                r["rs_starter_avg_ip"],
                r["rs_total_outs"] / (3.0 * r["rs_starts"]),
            ),
            axis=1,
        )
    ).sum())
    delta_formula_bad = int((
        matched.apply(
            lambda r: not close(
                r["delta_ps_minus_rs_ip"],
                r["actual_ip"] - r["rs_starter_avg_ip"],
            ),
            axis=1,
        )
    ).sum())
    unavailable_nonnull_baseline = int(unavailable["rs_starter_avg_ip"].notna().sum())
    unavailable_nonnull_delta = int(unavailable["delta_ps_minus_rs_ip"].notna().sum())

    checks = {
        "upstream_ps_validation_pass": ps_validation.get("status") == "PASS",
        "upstream_rs_validation_pass": rs_validation.get("status") == "PASS",
        "ps_rows_2016_2025_exactly_286": int(len(joined)) == 286,
        "ps_games_2016_2025_exactly_143": int(joined["game_pk"].nunique()) == 143,
        "eval_2017_2025_games_exactly_129": int(eval_all["game_pk"].nunique()) == 129,
        "eval_2017_2025_rows_exactly_258": int(len(eval_all)) == 258,
        "eval_2017_2025_matched_exactly_256": int(len(eval_matched)) == 256,
        "eval_2017_2025_unavailable_exactly_2": int(len(eval_unavailable)) == 2,
        "unavailable_identities_exact": observed_missing == EXPECTED_MISSING_2017_2025,
        "no_duplicate_rs_player_season_keys": duplicate_rs_keys == 0,
        "ps_ip_equals_outs_over_3": ps_ip_bad == 0,
        "rs_avg_ip_formula_exact": rs_avg_formula_bad == 0,
        "delta_formula_exact": delta_formula_bad == 0,
        "unavailable_has_no_baseline": unavailable_nonnull_baseline == 0,
        "unavailable_has_no_delta": unavailable_nonnull_delta == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    by_season = []
    for season, g in joined.groupby("season", sort=True):
        gm = g[g["rs_link_status"] == "MATCHED"]
        gu = g[g["rs_link_status"] != "MATCHED"]
        by_season.append({
            "season": int(season),
            "games": int(g["game_pk"].nunique()),
            "ps_starter_appearances": int(len(g)),
            "matched": int(len(gm)),
            "unavailable": int(len(gu)),
        })

    missing_details = [
        {
            "season": int(r.season),
            "game_date": str(r.game_date),
            "game_pk": str(r.game_pk),
            "team": str(r.team),
            "starter_id": str(r.starter_id),
            "starter_name": str(r.starter_name),
            "actual_ps_ip": float(r.actual_ip),
            "status": str(r.rs_link_status),
        }
        for r in unavailable.itertuples(index=False)
    ]

    report = {
        "status": status,
        "step": 3,
        "definition": "PS starter appearance joined to same-season player RS starter-average IP",
        "join_key": ["season", "Naver stable player id"],
        "delta_definition": "actual_PS_IP - same-season_RS_starter_avg_IP",
        "policy": {
            "name_match_fallback": "FORBIDDEN",
            "team_or_reliever_fallback": "FORBIDDEN",
            "zero_same_season_rs_starts": "UNAVAILABLE",
            "rolling_historical_shift": "NOT_COMPUTED_IN_STEP_3",
        },
        "coverage": {
            "ps_games_2016_2025": int(joined["game_pk"].nunique()),
            "ps_starter_appearances_2016_2025": int(len(joined)),
            "matched_2016_2025": int(len(matched)),
            "unavailable_2016_2025": int(len(unavailable)),
            "ps_games_2017_2025": int(eval_all["game_pk"].nunique()),
            "ps_starter_appearances_2017_2025": int(len(eval_all)),
            "matched_2017_2025": int(len(eval_matched)),
            "unavailable_2017_2025": int(len(eval_unavailable)),
        },
        "by_season": by_season,
        "unavailable_details": missing_details,
        "failure_counts": {
            "duplicate_rs_player_season_keys": duplicate_rs_keys,
            "bad_ps_ip_rows": ps_ip_bad,
            "bad_rs_avg_formula_rows": rs_avg_formula_bad,
            "bad_delta_formula_rows": delta_formula_bad,
            "unavailable_nonnull_baseline_rows": unavailable_nonnull_baseline,
            "unavailable_nonnull_delta_rows": unavailable_nonnull_delta,
        },
        "checks": checks,
        "inputs": {
            "ps": str(ps_path),
            "ps_validation": str(ps_validation_path),
            "rs": str(rs_path),
            "rs_validation": str(rs_validation_path),
        },
    }

    output_csv = out / "ps_rs_starter_avg_ip.csv"
    output_validation = out / "ps_rs_starter_avg_ip_validation.json"
    joined.sort_values(["season", "game_date", "game_pk", "home_away"], kind="stable").to_csv(
        output_csv, index=False
    )
    report["output"] = str(output_csv)
    output_validation.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Step 4 leakage-free expanding postseason Expected-IP shift table.

For target season Y:
  shift(Y) = mean(delta_ps_minus_rs_ip for postseason starter appearances in seasons < Y)

Only rows with a reproducible same-season RS starter baseline are eligible for the
mean. Rows with zero same-season RS starts stay UNAVAILABLE and are counted in the
audit; they are never imputed.

This script does NOT calculate target-season candidate predictions or RMSE/MAE.
Those belong to the next validation step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

TARGET_SEASONS = list(range(2017, 2027))
EXPECTED_HISTORY = {
    2017: (28, 28, 0),
    2018: (58, 58, 0),
    2019: (90, 89, 1),
    2020: (114, 113, 1),
    2021: (140, 139, 1),
    2022: (162, 161, 1),
    2023: (194, 192, 2),
    2024: (222, 220, 2),
    2025: (254, 252, 2),
    2026: (286, 284, 2),
}
FROZEN_2026_REFERENCE_SHIFT = -0.7588


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="research/postseason_expected_ip/ps_rs_starter_avg_ip.csv",
    )
    ap.add_argument(
        "--step3-validation",
        default="research/postseason_expected_ip/ps_rs_starter_avg_ip_validation.json",
    )
    ap.add_argument(
        "--frozen-reference",
        default="research/postseason_expected_ip/frozen_2026_prior.json",
    )
    ap.add_argument(
        "--output-dir",
        default="research/postseason_expected_ip",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    step3_path = Path(args.step3_validation)
    frozen_path = Path(args.frozen_reference)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, dtype={"starter_id": str, "game_pk": str})
    step3 = json.loads(step3_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    required = {
        "season",
        "game_pk",
        "starter_id",
        "starter_name",
        "actual_ip",
        "rs_link_status",
        "rs_starter_avg_ip",
        "delta_ps_minus_rs_ip",
    }
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    rows = []
    leakage_failures = []
    history_count_failures = []
    formula_failures = []

    for target in TARGET_SEASONS:
        history = df[df["season"] < target].copy()
        eligible = history[
            history["rs_link_status"].eq("MATCHED")
            & history["delta_ps_minus_rs_ip"].notna()
        ].copy()
        unavailable = history[~history.index.isin(eligible.index)].copy()

        if history.empty or eligible.empty:
            raise ValueError(f"target {target}: empty history/eligible history")

        if int(history["season"].max()) >= target:
            leakage_failures.append(target)

        expected = EXPECTED_HISTORY[target]
        got = (len(history), len(eligible), len(unavailable))
        if got != expected:
            history_count_failures.append(
                {"target_season": target, "got": got, "expected": expected}
            )

        shift = float(eligible["delta_ps_minus_rs_ip"].mean())
        direct_formula = float(
            (eligible["actual_ip"] - eligible["rs_starter_avg_ip"]).mean()
        )
        if not close(shift, direct_formula):
            formula_failures.append(target)

        rows.append(
            {
                "target_season": target,
                "source_start_season": int(history["season"].min()),
                "source_through_season": int(history["season"].max()),
                "historical_ps_games_total": int(history["game_pk"].nunique()),
                "historical_ps_starter_appearances_total": int(len(history)),
                "historical_delta_eligible_appearances": int(len(eligible)),
                "historical_unavailable_appearances": int(len(unavailable)),
                "historical_ps_mean_ip_shift": shift,
                "min_delta_ip": float(eligible["delta_ps_minus_rs_ip"].min()),
                "max_delta_ip": float(eligible["delta_ps_minus_rs_ip"].max()),
                "mean_actual_ps_ip": float(eligible["actual_ip"].mean()),
                "mean_same_season_rs_starter_avg_ip": float(
                    eligible["rs_starter_avg_ip"].mean()
                ),
            }
        )

    shifts = pd.DataFrame(rows)

    target_sequence_ok = shifts["target_season"].astype(int).tolist() == TARGET_SEASONS
    source_through_ok = bool(
        (shifts["source_through_season"].astype(int) == shifts["target_season"].astype(int) - 1).all()
    )
    source_start_ok = bool((shifts["source_start_season"].astype(int) == 2016).all())
    cumulative_counts_monotone = bool(
        shifts["historical_delta_eligible_appearances"].is_monotonic_increasing
    )

    s2026 = shifts.loc[shifts["target_season"] == 2026].iloc[0]
    rebuilt_2026_shift = float(s2026["historical_ps_mean_ip_shift"])
    frozen_shift = float(
        frozen.get("research_basis", {}).get(
            "mean_shift_ip", FROZEN_2026_REFERENCE_SHIFT
        )
    )
    frozen_residual_appearances = int(
        frozen.get("research_basis", {}).get("residual_appearances", 286)
    )

    checks = {
        "step3_validation_pass": step3.get("status") == "PASS",
        "target_seasons_exactly_2017_2026": target_sequence_ok,
        "source_starts_at_2016_for_every_target": source_start_ok,
        "source_through_is_target_minus_one": source_through_ok,
        "no_target_or_future_season_leakage": len(leakage_failures) == 0,
        "historical_counts_exact": len(history_count_failures) == 0,
        "shift_formula_exact": len(formula_failures) == 0,
        "eligible_counts_monotone": cumulative_counts_monotone,
        "target_2026_total_history_exactly_286": int(s2026["historical_ps_starter_appearances_total"]) == 286,
        "target_2026_eligible_history_exactly_284": int(s2026["historical_delta_eligible_appearances"]) == 284,
        "target_2026_unavailable_history_exactly_2": int(s2026["historical_unavailable_appearances"]) == 2,
        "step4_does_not_emit_prediction_metrics": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    validation = {
        "status": status,
        "step": 4,
        "definition": "Leakage-free expanding historical PS IP shift by target season",
        "formula": "mean(delta_ps_minus_rs_ip where season < target_season and RS link is MATCHED)",
        "policy": {
            "target_or_future_ps_data": "FORBIDDEN",
            "unavailable_delta_imputation": "FORBIDDEN",
            "zero_same_season_rs_start_history": "EXCLUDED_FROM_SHIFT_AND_AUDITED",
            "candidate_predictions_and_rmse": "NOT_COMPUTED_IN_STEP_4",
        },
        "targets": TARGET_SEASONS,
        "checks": checks,
        "failure_details": {
            "leakage_targets": leakage_failures,
            "history_count_failures": history_count_failures,
            "formula_failure_targets": formula_failures,
        },
        "rebuilt_2026": {
            "historical_ps_starter_appearances_total": int(s2026["historical_ps_starter_appearances_total"]),
            "eligible_delta_appearances": int(s2026["historical_delta_eligible_appearances"]),
            "unavailable_delta_appearances": int(s2026["historical_unavailable_appearances"]),
            "mean_shift_ip": rebuilt_2026_shift,
        },
        "frozen_2026_reference": {
            "mean_shift_ip": frozen_shift,
            "residual_appearances_claimed": frozen_residual_appearances,
            "rebuilt_minus_frozen_shift_ip": rebuilt_2026_shift - frozen_shift,
            "interpretation": "NON_BINDING_FORENSIC_REFERENCE; old file claims 286 residuals, rebuilt reproducible table has 284 eligible deltas",
        },
        "inputs": {
            "joined_step3": str(inp),
            "step3_validation": str(step3_path),
            "frozen_reference": str(frozen_path),
        },
        "output": str(out / "ps_expected_ip_expanding_shift_by_target_season.csv"),
    }

    shifts.to_csv(out / "ps_expected_ip_expanding_shift_by_target_season.csv", index=False)
    (out / "ps_expected_ip_expanding_shift_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(validation, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import compare_mu_v1_2_bp30_discrepancy as cmp

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "research" / "mu-v1.2" / "bp30_prospective_promotion_policy.json"
DEFAULT_GLOB = "state/prospective_mu_v1_2/*.json"
DEFAULT_OUT = "state/mu_v1_2_bp30_prospective_promotion_gate.json"


def _load_snapshot(path: Path) -> dict:
    snap = json.loads(path.read_text(encoding="utf-8"))
    if snap.get("capture_type") != "PROSPECTIVE_CLEAN_PRE_RESEARCH":
        raise RuntimeError(f"ineligible snapshot capture_type: {path}")
    if snap.get("market_state", {}).get("prices_inspected") is not False:
        raise RuntimeError(f"ineligible snapshot market state: {path}")
    if "input_payload" not in snap:
        raise RuntimeError(f"snapshot missing input_payload: {path}")
    return snap


def evaluate(snapshot_paths: list[Path], policy: dict) -> dict:
    fields = list(policy["score_delta_gate"]["eligible_fields"])
    deltas: list[float] = []
    signed: list[float] = []
    dates: set[str] = set()
    teams: set[str] = set()
    game_count = 0
    side_flips = 0
    material_margin_flips = 0
    rows = []

    for path in snapshot_paths:
        snap = _load_snapshot(path)
        comp = cmp.compare_payload(snap["input_payload"])
        dates.add(str(comp["target_date"]))
        for g in comp["games"]:
            game_count += 1
            teams.update([str(g["away_team"]), str(g["home_team"])])
            a = g["active_prediction"]
            b = g["calendar30_p39_25_prediction"]
            field_rows = []
            for f in fields:
                d = float(b[f]) - float(a[f])
                signed.append(d)
                deltas.append(abs(d))
                field_rows.append({"field": f, "delta_alt_minus_active": d, "abs_delta": abs(d)})

            active_margin = float(a["home_full"] if "home_full" in a else a["full_home"]) - float(a["away_full"] if "away_full" in a else a["full_away"])
            alt_margin = float(b["home_full"] if "home_full" in b else b["full_home"]) - float(b["away_full"] if "away_full" in b else b["full_away"])
            flip = (active_margin > 0 and alt_margin < 0) or (active_margin < 0 and alt_margin > 0)
            if flip:
                side_flips += 1
                if abs(active_margin) >= float(policy["decision_stability_gate"]["material_active_full_mu_margin_runs"]):
                    material_margin_flips += 1
            try:
                source_snapshot = str(path.relative_to(ROOT))
            except ValueError:
                source_snapshot = str(path)
            rows.append({
                "source_snapshot": source_snapshot,
                "target_date": comp["target_date"],
                "game_pk": g.get("game_pk"),
                "away_team": g["away_team"],
                "home_team": g["home_team"],
                "active_full_mu_margin_home_minus_away": active_margin,
                "alt_full_mu_margin_home_minus_away": alt_margin,
                "predicted_side_flip": bool(flip),
                "field_deltas": field_rows,
            })

    req = policy["required_prospective_sample"]
    sample_checks = {
        "minimum_games": game_count >= int(req["minimum_games"]),
        "minimum_distinct_capture_dates": len(dates) >= int(req["minimum_distinct_capture_dates"]),
        "minimum_distinct_teams": len(teams) >= int(req["minimum_distinct_teams"]),
    }
    sample_ok = all(sample_checks.values())

    if deltas:
        median_abs = float(np.median(deltas))
        p95_abs = float(np.quantile(deltas, 0.95))
        max_abs = float(np.max(deltas))
        mean_signed = float(np.mean(signed))
    else:
        median_abs = p95_abs = max_abs = mean_signed = None

    lim = policy["score_delta_gate"]["absolute_delta_runs"]
    bias_lim = float(policy["score_delta_gate"]["signed_bias"]["mean_signed_delta_abs_max"])
    numeric_checks = {
        "median_abs_delta": median_abs is not None and median_abs <= float(lim["median_max"]),
        "p95_abs_delta": p95_abs is not None and p95_abs <= float(lim["p95_max"]),
        "max_abs_delta": max_abs is not None and max_abs <= float(lim["single_observation_max"]),
        "mean_signed_bias": mean_signed is not None and abs(mean_signed) <= bias_lim,
    }
    numeric_ok = all(numeric_checks.values())

    flip_rate = float(side_flips / game_count) if game_count else None
    ds = policy["decision_stability_gate"]
    decision_checks = {
        "side_flip_rate": flip_rate is not None and flip_rate <= float(ds["predicted_side_flip_rate_max"]),
        "no_material_margin_flip": material_margin_flips == 0,
    }
    decision_ok = all(decision_checks.values())

    if not sample_ok:
        verdict = "WAIT_INSUFFICIENT_PROSPECTIVE_SAMPLE"
    elif numeric_ok and decision_ok:
        verdict = "PASS_PRACTICAL_DISCREPANCY_BOUND"
    else:
        verdict = "FAIL_PRACTICAL_DISCREPANCY_BOUND"

    return {
        "version": "mu-v1.2-bp30-prospective-promotion-gate-evaluation-v0.1",
        "policy_version": policy["version"],
        "verdict": verdict,
        "official_shadow_allowed": False,
        "note": "A PASS only clears the BP30 practical-discrepancy sub-gate. It does not by itself activate Official Mechanical Shadow.",
        "sample": {
            "snapshot_files": len(snapshot_paths),
            "games": game_count,
            "distinct_capture_dates": len(dates),
            "distinct_teams": len(teams),
            "dates": sorted(dates),
            "teams": sorted(teams),
            "checks": sample_checks,
        },
        "score_delta_summary": {
            "eligible_observations": len(deltas),
            "median_abs_delta_runs": median_abs,
            "p95_abs_delta_runs": p95_abs,
            "max_abs_delta_runs": max_abs,
            "mean_signed_delta_alt_minus_active": mean_signed,
            "checks": numeric_checks,
        },
        "decision_stability": {
            "predicted_side_flips": side_flips,
            "side_flip_rate": flip_rate,
            "material_margin_flips": material_margin_flips,
            "checks": decision_checks,
        },
        "games": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    paths = [Path(p) for p in sorted(glob.glob(str(ROOT / args.snapshot_glob)))]
    result = evaluate(paths, policy)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        out_label = str(out.relative_to(ROOT))
    except ValueError:
        out_label = str(out)
    print(json.dumps({
        "verdict": result["verdict"],
        "games": result["sample"]["games"],
        "dates": result["sample"]["distinct_capture_dates"],
        "teams": result["sample"]["distinct_teams"],
        "out": out_label,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import mu_v1_2_recovered_challenger as active
import mu_v1_3 as engine

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
ALT_WINDOW_DAYS = 30
ALT_PSEUDO_OUTS = 39.25


def _alt_bp30_calendar30_p39_25(bp6, bprel, *, d, opponent):
    """Earlier pregame reliever-stint candidate retained only as discrepancy bound.

    This is NOT a retuning candidate. It exists to quantify how much the unresolved
    historical label mismatch (feature name '30d' vs active forensic calendar36)
    changes live features/predictions while every other recovered semantic is fixed.
    """
    d = pd.Timestamp(d).normalize()
    opp = str(opponent)

    pre_all = bp6[bp6.game_date < d]
    pre = pre_all[pre_all.game_date.dt.year == d.year]
    if pre.empty:
        pre = pre_all

    lg_long = active._safe_ra9(float(pre.runs.sum()), float(pre.outs.sum()), 4.5)
    opp_long = pre[pre.pitching_team.astype(str).eq(opp)]
    lr = float(opp_long.runs.sum())
    lo = float(opp_long.outs.sum())
    team_long = (
        (lr + active.BP_LONG_OUTS_P * lg_long / 27.0) * 27.0 / (lo + active.BP_LONG_OUTS_P)
        if lo + active.BP_LONG_OUTS_P > 0 else lg_long
    )

    rec = bprel[(bprel.game_date < d) & (bprel.game_date >= d - pd.Timedelta(days=ALT_WINDOW_DAYS))]
    opp_rec = rec[rec.pitching_team.astype(str).eq(opp)]
    lg_recent = active._safe_ra9(float(rec.runs.sum()), float(rec.outs.sum()), lg_long)
    rr = float(opp_rec.runs.sum())
    ro = float(opp_rec.outs.sum())
    team_recent = (
        (rr + ALT_PSEUDO_OUTS * team_long / 27.0) * 27.0 / (ro + ALT_PSEUDO_OUTS)
        if ro + ALT_PSEUDO_OUTS > 0 else team_long
    )
    return float(team_recent / lg_recent) if lg_recent else 1.0


def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, p))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _prediction_delta(active_pred: dict, alt_pred: dict) -> dict:
    a = _flatten_numeric(active_pred)
    b = _flatten_numeric(alt_pred)
    common = sorted(set(a) & set(b))
    rows = [
        {"field": k, "active": a[k], "calendar30_p39_25": b[k], "delta_alt_minus_active": b[k] - a[k], "abs_delta": abs(b[k] - a[k])}
        for k in common
    ]
    rows.sort(key=lambda r: r["abs_delta"], reverse=True)
    return {
        "max_abs_numeric_delta": rows[0]["abs_delta"] if rows else None,
        "largest_numeric_deltas": rows[:20],
    }


def compare_payload(payload: dict) -> dict:
    target = pd.Timestamp(payload["target_date"]).normalize()
    tg, st, bp6, bprel = active._load_history(target)
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    games_out = []
    for g in payload.get("games", []):
        sides = {}
        for team, opponent, starter, is_home in [
            (str(g["away_team"]), str(g["home_team"]), str(g["away_starter"]), 0),
            (str(g["home_team"]), str(g["away_team"]), str(g["home_starter"]), 1),
        ]:
            f_active = active.build_pregame_features(
                tg, st, bp6, bprel,
                d=target,
                game_pk=str(g.get("game_pk", "UNKNOWN")),
                team=team,
                opponent=opponent,
                self_starter=starter,
                is_home=is_home,
                park_factor_override=g.get("park_factor"),
            )
            f_alt = dict(f_active)
            f_alt["opp_bp_ra9_30d_rel"] = _alt_bp30_calendar30_p39_25(
                bp6, bprel, d=target, opponent=opponent
            )
            sides["home" if is_home else "away"] = {
                "team": team,
                "opponent": opponent,
                "active_bp30_calendar36_p25_5": float(f_active["opp_bp_ra9_30d_rel"]),
                "alt_bp30_calendar30_p39_25": float(f_alt["opp_bp_ra9_30d_rel"]),
                "feature_delta_alt_minus_active": float(f_alt["opp_bp_ra9_30d_rel"] - f_active["opp_bp_ra9_30d_rel"]),
                "active_features": f_active,
                "alt_features": f_alt,
            }

        pred_active = engine.score(model, sides["away"]["active_features"], sides["home"]["active_features"])
        pred_alt = engine.score(model, sides["away"]["alt_features"], sides["home"]["alt_features"])
        games_out.append({
            "game_pk": g.get("game_pk"),
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "away_bp30": {k: v for k, v in sides["away"].items() if not k.endswith("features")},
            "home_bp30": {k: v for k, v in sides["home"].items() if not k.endswith("features")},
            "active_prediction": pred_active,
            "calendar30_p39_25_prediction": pred_alt,
            "prediction_delta": _prediction_delta(pred_active, pred_alt),
        })

    all_max = [g["prediction_delta"]["max_abs_numeric_delta"] for g in games_out if g["prediction_delta"]["max_abs_numeric_delta"] is not None]
    return {
        "version": "mu-v1.2-bp30-live-discrepancy-bound-v0.1",
        "status": "DIAGNOSTIC_ONLY_NOT_MODEL_SELECTION",
        "target_date": str(target.date()),
        "active_semantic": "RELIEVER_STINT_PREGAME_CALENDAR36_P25.5",
        "comparison_semantic": "RELIEVER_STINT_PREGAME_CALENDAR30_P39.25",
        "discipline": "Comparison exists only to bound practical prediction impact of the unresolved historical 30d-label/calendar36 discrepancy; do not select or retune from prospective outcomes.",
        "max_abs_numeric_prediction_delta_across_games": max(all_max) if all_max else None,
        "games": games_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Raw pregame input JSON. Prefer --snapshot for genuine prospective audit.")
    ap.add_argument("--snapshot", help="Captured prospective snapshot JSON from state/prospective_mu_v1_2/.")
    ap.add_argument("--out", default="state/mu_v1_2_bp30_live_discrepancy_bound.json")
    args = ap.parse_args()

    if bool(args.input) == bool(args.snapshot):
        raise SystemExit("Pass exactly one of --input or --snapshot")

    if args.snapshot:
        snap = json.loads((ROOT / args.snapshot).read_text(encoding="utf-8"))
        if snap.get("capture_type") != "PROSPECTIVE_CLEAN_PRE_RESEARCH":
            raise SystemExit("Snapshot is not a PROSPECTIVE_CLEAN_PRE_RESEARCH artifact")
        if snap.get("market_state", {}).get("prices_inspected") is not False:
            raise SystemExit("Snapshot is not marked market-unseen")
        payload = snap["input_payload"]
        source = args.snapshot
    else:
        payload = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
        source = args.input

    out = compare_payload(payload)
    out["source"] = source
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(p.relative_to(ROOT)),
        "target_date": out["target_date"],
        "games": len(out["games"]),
        "max_abs_numeric_prediction_delta_across_games": out["max_abs_numeric_prediction_delta_across_games"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

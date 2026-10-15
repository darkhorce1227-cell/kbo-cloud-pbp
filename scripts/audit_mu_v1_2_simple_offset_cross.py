from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_simple_offset_semantics as base

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_simple_offset_cross.json"

MODES = [
    "calendar30_strict",
    "calendar30_exclusive_lower",
    "season_calendar30_strict",
    "last30_league_games",
    "last150_league_games",
    "calendar30_same_day",
]


def offset_vectors(rows: pd.DataFrame, tg: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(rows.game_date).dt.normalize().tolist()
    mp = base.league_offsets(tg, dates, mode)
    f5, late = [], []
    for d0 in pd.to_datetime(rows.game_date).dt.normalize():
        a, b = mp[pd.Timestamp(d0)]
        f5.append(a)
        late.append(b)
    return np.asarray(f5, float), np.asarray(late, float)


def mean_mu(rows: pd.DataFrame, model: dict) -> dict[str, float]:
    f5, full = [], []
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away = x[x.is_home == 0].iloc[0].to_dict()
        home = x[x.is_home == 1].iloc[0].to_dict()
        p = engine.score(model, away, home)
        f5.extend([float(p["f5_away"]), float(p["f5_home"])])
        full.extend([float(p["full_away"]), float(p["full_home"])])
    return {"f5": float(np.mean(f5)), "full": float(np.mean(full))}


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    dev = coeff.make_dev_rows(tg, st, bp6)
    oos = recovered.make_oos_rows(tg, st, bp6)
    if len(dev) != 2628 or dev.game_pk.nunique() != 1314:
        raise RuntimeError(f"dev gate {len(dev)}/{dev.game_pk.nunique()}")
    if len(oos) != 1368 or oos.game_pk.nunique() != 684:
        raise RuntimeError(f"oos gate {len(oos)}/{oos.game_pk.nunique()}")

    dev_vec = {m: offset_vectors(dev, tg, m) for m in MODES}
    oos_vec = {m: offset_vectors(oos, tg, m) for m in MODES}

    f5_fit = {}
    late_fit = {}
    for mode in MODES:
        d = dev.copy()
        d["league_f5_30d"] = dev_vec[mode][0]
        f5_fit[mode] = coeff.fit_component(
            d, model, "simple_f5", engine.SIMPLE_F5,
            "runs_f5", "league_f5_30d", 1.0,
        )
        d2 = dev.copy()
        d2["league_late_30d"] = dev_vec[mode][1]
        late_fit[mode] = coeff.fit_component(
            d2, model, "simple_late", engine.SIMPLE_LATE,
            "runs_late_6plus", "league_late_30d", 1.0,
        )

    pub_f5 = float(metrics["oos_2026"]["f5"]["hybrid"]["mean_mu"])
    pub_full = float(metrics["oos_2026"]["full"]["hybrid"]["mean_mu"])
    results = []
    for fm in MODES:
        for lm in MODES:
            oo = oos.copy()
            oo["league_f5_30d"] = oos_vec[fm][0]
            oo["league_late_30d"] = oos_vec[lm][1]
            mm = mean_mu(oo, model)
            fp = f5_fit[fm]
            lp = late_fit[lm]
            results.append({
                "f5_offset_mode": fm,
                "late_offset_mode": lm,
                "simple_f5_param_norm_mae": float(fp["param_norm_mae"]),
                "simple_f5_param_corr": float(fp["param_corr"]),
                "simple_late_param_norm_mae": float(lp["param_norm_mae"]),
                "simple_late_param_corr": float(lp["param_corr"]),
                "simple_sum_param_norm": float(fp["param_norm_mae"] + lp["param_norm_mae"]),
                "oos_mean_mu_f5": mm["f5"],
                "oos_mean_mu_full": mm["full"],
                "delta_published_f5_mean_mu": float(mm["f5"] - pub_f5),
                "delta_published_full_mean_mu": float(mm["full"] - pub_full),
                "oos_total_gap_abs_sum": float(abs(mm["f5"] - pub_f5) + abs(mm["full"] - pub_full)),
            })

    by_coeff = sorted(results, key=lambda x: x["simple_sum_param_norm"])
    by_oos = sorted(results, key=lambda x: x["oos_total_gap_abs_sum"])
    # Pareto frontier: no other candidate is better on both frozen coefficient parity and OOS total gap.
    pareto = []
    for r in results:
        dominated = any(
            (q["simple_sum_param_norm"] <= r["simple_sum_param_norm"])
            and (q["oos_total_gap_abs_sum"] <= r["oos_total_gap_abs_sum"])
            and (
                q["simple_sum_param_norm"] < r["simple_sum_param_norm"]
                or q["oos_total_gap_abs_sum"] < r["oos_total_gap_abs_sum"]
            )
            for q in results
        )
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x["simple_sum_param_norm"])

    payload = {
        "version": "mu-v1.2-simple-offset-cross-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "oos_rows": int(len(oos)),
        "oos_games": int(oos.game_pk.nunique()),
        "published_mean_mu": {"f5": pub_f5, "full": pub_full},
        "modes": MODES,
        "ranked_by_frozen_simple_coefficient_parity": by_coeff,
        "pareto_frontier_coeff_vs_oos_mean_mu": pareto,
        "ranked_by_2026_mean_mu_match_posthoc_only": by_oos,
        "decision_rule": "Historical choice must be led by 2024-2025 frozen simple coefficient parity. 2026 mean-mu is secondary and may identify tension, not select the historical formula by itself.",
        "note": "F5 and late league offsets are crossed independently. Historical late training target remains inning-6-plus PA runs.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

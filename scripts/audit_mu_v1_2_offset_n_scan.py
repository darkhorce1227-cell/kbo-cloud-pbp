from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_coefficient_parity as coeff

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_offset_n_scan.json"
NS = list(range(100, 201, 5))


def offsets_for_n(tg: pd.DataFrame, dates: list[pd.Timestamp], n_games: int) -> dict[pd.Timestamp, tuple[float, float]]:
    out = {}
    games = tg[["game_date", "game_pk"]].drop_duplicates().sort_values(["game_date", "game_pk"])
    for d0 in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
        d = pd.Timestamp(d0).normalize()
        ids = games[games.game_date < d].tail(int(n_games)).game_pk.astype(str).tolist()
        h = tg[tg.game_pk.astype(str).isin(ids)]
        if h.empty:
            raise RuntimeError(f"empty history for {d.date()} n={n_games}")
        out[d] = (float(h.runs_f5.mean()), float(h.late_6plus_pa.mean()))
    return out


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    dev = coeff.make_dev_rows(tg, st, bp6)
    if len(dev) != 2628 or dev.game_pk.nunique() != 1314:
        raise RuntimeError(f"dev gate {len(dev)}/{dev.game_pk.nunique()}")
    dates = pd.to_datetime(dev.game_date).dt.normalize().tolist()
    dnorm = pd.to_datetime(dev.game_date).dt.normalize()

    rows = []
    for n in NS:
        mp = offsets_for_n(tg, dates, n)
        f5 = np.asarray([mp[pd.Timestamp(d)][0] for d in dnorm], float)
        late = np.asarray([mp[pd.Timestamp(d)][1] for d in dnorm], float)

        a = dev.copy(); a["league_f5_30d"] = f5
        b = dev.copy(); b["league_late_30d"] = late
        ff = coeff.fit_component(a, model, "simple_f5", engine.SIMPLE_F5, "runs_f5", "league_f5_30d", 1.0)
        lf = coeff.fit_component(b, model, "simple_late", engine.SIMPLE_LATE, "runs_late_6plus", "league_late_30d", 1.0)
        rows.append({
            "n_league_games": int(n),
            "simple_f5_param_norm_mae": float(ff["param_norm_mae"]),
            "simple_f5_param_corr": float(ff["param_corr"]),
            "simple_late_param_norm_mae": float(lf["param_norm_mae"]),
            "simple_late_param_corr": float(lf["param_corr"]),
            "simple_sum_param_norm": float(ff["param_norm_mae"] + lf["param_norm_mae"]),
        })

    payload = {
        "version": "mu-v1.2-offset-n-scan-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "scan": {"min": min(NS), "max": max(NS), "step": 5},
        "best_f5": min(rows, key=lambda x: x["simple_f5_param_norm_mae"]),
        "best_late": min(rows, key=lambda x: x["simple_late_param_norm_mae"]),
        "best_combined": min(rows, key=lambda x: x["simple_sum_param_norm"]),
        "top10_f5": sorted(rows, key=lambda x: x["simple_f5_param_norm_mae"])[:10],
        "top10_late": sorted(rows, key=lambda x: x["simple_late_param_norm_mae"])[:10],
        "top10_combined": sorted(rows, key=lambda x: x["simple_sum_param_norm"])[:10],
        "all": rows,
        "decision_rule": "This scan uses 2024-2025 frozen simple coefficient parity only. It does not inspect 2026 outcomes to select N.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

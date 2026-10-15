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
OUT_PATH = ROOT / "state" / "mu_v1_2_offset_n_refine.json"
F5_NS = list(range(120, 141))
LATE_NS = list(range(180, 301, 5))


def offsets_for_n(tg: pd.DataFrame, dates: list[pd.Timestamp], n_games: int) -> tuple[np.ndarray, np.ndarray]:
    games = tg[["game_date", "game_pk"]].drop_duplicates().sort_values(["game_date", "game_pk"])
    f5_map, late_map = {}, {}
    for d0 in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
        d = pd.Timestamp(d0).normalize()
        ids = games[games.game_date < d].tail(int(n_games)).game_pk.astype(str).tolist()
        h = tg[tg.game_pk.astype(str).isin(ids)]
        if h.empty:
            raise RuntimeError(f"empty history for {d.date()} n={n_games}")
        f5_map[d] = float(h.runs_f5.mean())
        late_map[d] = float(h.late_6plus_pa.mean())
    dnorm = pd.to_datetime(dates).normalize()
    return (
        np.asarray([f5_map[pd.Timestamp(d)] for d in dnorm], float),
        np.asarray([late_map[pd.Timestamp(d)] for d in dnorm], float),
    )


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    dev = coeff.make_dev_rows(tg, st, bp6)
    if len(dev) != 2628 or dev.game_pk.nunique() != 1314:
        raise RuntimeError(f"dev gate {len(dev)}/{dev.game_pk.nunique()}")
    dates = pd.to_datetime(dev.game_date).dt.normalize().tolist()

    f5_rows = []
    for n in F5_NS:
        f5, _ = offsets_for_n(tg, dates, n)
        d = dev.copy(); d["league_f5_30d"] = f5
        fit = coeff.fit_component(d, model, "simple_f5", engine.SIMPLE_F5, "runs_f5", "league_f5_30d", 1.0)
        f5_rows.append({
            "n_league_games": int(n),
            "param_norm_mae": float(fit["param_norm_mae"]),
            "param_corr": float(fit["param_corr"]),
            "param_mae": float(fit["param_mae"]),
        })

    late_rows = []
    for n in LATE_NS:
        _, late = offsets_for_n(tg, dates, n)
        d = dev.copy(); d["league_late_30d"] = late
        fit = coeff.fit_component(d, model, "simple_late", engine.SIMPLE_LATE, "runs_late_6plus", "league_late_30d", 1.0)
        late_rows.append({
            "n_league_games": int(n),
            "param_norm_mae": float(fit["param_norm_mae"]),
            "param_corr": float(fit["param_corr"]),
            "param_mae": float(fit["param_mae"]),
        })

    payload = {
        "version": "mu-v1.2-offset-n-refine-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "f5_scan": {"min": min(F5_NS), "max": max(F5_NS), "step": 1},
        "late_scan": {"min": min(LATE_NS), "max": max(LATE_NS), "step": 5},
        "best_f5": min(f5_rows, key=lambda x: x["param_norm_mae"]),
        "best_late": min(late_rows, key=lambda x: x["param_norm_mae"]),
        "top10_f5": sorted(f5_rows, key=lambda x: x["param_norm_mae"])[:10],
        "top10_late": sorted(late_rows, key=lambda x: x["param_norm_mae"])[:10],
        "f5_all": f5_rows,
        "late_all": late_rows,
        "decision_rule": "Frozen 2024-2025 simple coefficient parity only; no 2026 outcome selection.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

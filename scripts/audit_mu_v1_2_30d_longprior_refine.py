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
OUT_PATH = ROOT / "state" / "mu_v1_2_30d_longprior_refine.json"
PS = list(range(0, 41))


def precompute(tg: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, tuple[float, float, float, float, float]]:
    out = {}
    for d0 in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
        d = pd.Timestamp(d0).normalize()
        pre = tg[tg.game_date < d]
        if pre.empty:
            raise RuntimeError(f"empty pre history at {d.date()}")
        h30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
        # Match the successful parent smoothing audit at offseason/opening boundaries:
        # when no game exists in the literal prior 30 calendar days, seed from the
        # most recent pregame history rather than leaking same-day games.
        if h30.empty:
            h30 = pre.tail(min(len(pre), 300))
        out[d] = (
            float(len(h30)),
            float(h30.runs_f5.sum()),
            float(h30.late_6plus_pa.sum()),
            float(pre.runs_f5.mean()),
            float(pre.late_6plus_pa.mean()),
        )
    return out


def vectors(dates: list[pd.Timestamp], stats, p: float) -> tuple[np.ndarray, np.ndarray]:
    f5, late = [], []
    for d0 in pd.to_datetime(dates).normalize():
        n, sf5, slate, lf5, llate = stats[pd.Timestamp(d0)]
        f5.append((sf5 + p * lf5) / (n + p))
        late.append((slate + p * llate) / (n + p))
    return np.asarray(f5, float), np.asarray(late, float)


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    dev = coeff.make_dev_rows(tg, st, bp6)
    if len(dev) != 2628 or dev.game_pk.nunique() != 1314:
        raise RuntimeError(f"dev gate {len(dev)}/{dev.game_pk.nunique()}")
    dates = pd.to_datetime(dev.game_date).dt.normalize().tolist()
    stats = precompute(tg, dates)

    rows = []
    for p in PS:
        f5, late = vectors(dates, stats, float(p))
        d1 = dev.copy(); d1["league_f5_30d"] = f5
        ff = coeff.fit_component(d1, model, "simple_f5", engine.SIMPLE_F5, "runs_f5", "league_f5_30d", 1.0)
        d2 = dev.copy(); d2["league_late_30d"] = late
        lf = coeff.fit_component(d2, model, "simple_late", engine.SIMPLE_LATE, "runs_late_6plus", "league_late_30d", 1.0)
        rows.append({
            "pseudo_team_games": int(p),
            "simple_f5_param_norm_mae": float(ff["param_norm_mae"]),
            "simple_f5_param_corr": float(ff["param_corr"]),
            "simple_late_param_norm_mae": float(lf["param_norm_mae"]),
            "simple_late_param_corr": float(lf["param_corr"]),
            "simple_sum_param_norm": float(ff["param_norm_mae"] + lf["param_norm_mae"]),
        })

    payload = {
        "version": "mu-v1.2-30d-longprior-refine-v0.2",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "formula": "calendar30 pooled team-game mean shrunk toward all-pregame-history league mean with P pseudo team-games; offseason empty-30D fallback uses most recent 300 pregame team-rows",
        "scan": {"min": min(PS), "max": max(PS), "step": 1},
        "best_f5": min(rows, key=lambda x: x["simple_f5_param_norm_mae"]),
        "best_late": min(rows, key=lambda x: x["simple_late_param_norm_mae"]),
        "best_combined": min(rows, key=lambda x: x["simple_sum_param_norm"]),
        "top10_f5": sorted(rows, key=lambda x: x["simple_f5_param_norm_mae"])[:10],
        "top10_late": sorted(rows, key=lambda x: x["simple_late_param_norm_mae"])[:10],
        "top10_combined": sorted(rows, key=lambda x: x["simple_sum_param_norm"])[:10],
        "all": rows,
        "decision_rule": "Use frozen 2024-2025 simple coefficient parity only. Prefer a stable/simple pseudo-count shared across F5 and late if supported; do not choose by 2026 OOS.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

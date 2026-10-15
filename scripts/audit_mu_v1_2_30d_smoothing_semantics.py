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
OUT_PATH = ROOT / "state" / "mu_v1_2_30d_smoothing_semantics.json"

PSEUDOS = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0]
HALFLIVES = [5.0, 10.0, 15.0, 20.0, 30.0]


def _mean_or(x: pd.Series, fallback: float) -> float:
    return float(x.mean()) if len(x) else float(fallback)


def _weighted_mean(vals: np.ndarray, w: np.ndarray, fallback: float) -> float:
    den = float(np.sum(w))
    return float(np.sum(vals * w) / den) if den > 0 else float(fallback)


def offsets_for_dates(tg: pd.DataFrame, dates: list[pd.Timestamp], mode: str) -> tuple[np.ndarray, np.ndarray]:
    f5_map: dict[pd.Timestamp, float] = {}
    late_map: dict[pd.Timestamp, float] = {}

    for d0 in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
        d = pd.Timestamp(d0).normalize()
        pre = tg[tg.game_date < d]
        if pre.empty:
            raise RuntimeError(f"empty pre history at {d.date()}")
        h30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
        if h30.empty:
            h30 = pre.tail(min(len(pre), 300))

        long_f5 = float(pre.runs_f5.mean())
        long_late = float(pre.late_6plus_pa.mean())
        season = pre[pre.game_date.dt.year == d.year]
        season_f5 = _mean_or(season.runs_f5, long_f5)
        season_late = _mean_or(season.late_6plus_pa, long_late)
        raw_f5 = float(h30.runs_f5.mean())
        raw_late = float(h30.late_6plus_pa.mean())

        if mode == "pooled_calendar30_raw":
            f5, late = raw_f5, raw_late
        elif mode == "equal_team_calendar30":
            f5 = float(h30.groupby("team").runs_f5.mean().mean())
            late = float(h30.groupby("team").late_6plus_pa.mean().mean())
        elif mode == "equal_day_calendar30":
            f5 = float(h30.groupby("game_date").runs_f5.mean().mean())
            late = float(h30.groupby("game_date").late_6plus_pa.mean().mean())
        elif mode.startswith("shrink_long_p"):
            p = float(mode.rsplit("p", 1)[1])
            n = float(len(h30))
            f5 = float((h30.runs_f5.sum() + p * long_f5) / (n + p))
            late = float((h30.late_6plus_pa.sum() + p * long_late) / (n + p))
        elif mode.startswith("shrink_season_p"):
            p = float(mode.rsplit("p", 1)[1])
            n = float(len(h30))
            f5 = float((h30.runs_f5.sum() + p * season_f5) / (n + p))
            late = float((h30.late_6plus_pa.sum() + p * season_late) / (n + p))
        elif mode.startswith("exp30_h"):
            hl = float(mode.rsplit("h", 1)[1])
            age = (d - h30.game_date).dt.days.to_numpy(float)
            w = np.power(0.5, age / hl)
            f5 = _weighted_mean(h30.runs_f5.to_numpy(float), w, raw_f5)
            late = _weighted_mean(h30.late_6plus_pa.to_numpy(float), w, raw_late)
        else:
            raise ValueError(mode)

        f5_map[d] = f5
        late_map[d] = late

    dnorm = pd.to_datetime(dates).normalize()
    return (
        np.asarray([f5_map[pd.Timestamp(d)] for d in dnorm], float),
        np.asarray([late_map[pd.Timestamp(d)] for d in dnorm], float),
    )


def fit_mode(dev: pd.DataFrame, tg: pd.DataFrame, model: dict, mode: str) -> dict:
    dates = pd.to_datetime(dev.game_date).dt.normalize().tolist()
    f5, late = offsets_for_dates(tg, dates, mode)

    d1 = dev.copy()
    d1["league_f5_30d"] = f5
    ff = coeff.fit_component(
        d1, model, "simple_f5", engine.SIMPLE_F5,
        "runs_f5", "league_f5_30d", 1.0,
    )

    d2 = dev.copy()
    d2["league_late_30d"] = late
    lf = coeff.fit_component(
        d2, model, "simple_late", engine.SIMPLE_LATE,
        "runs_late_6plus", "league_late_30d", 1.0,
    )

    return {
        "mode": mode,
        "simple_f5_param_norm_mae": float(ff["param_norm_mae"]),
        "simple_f5_param_corr": float(ff["param_corr"]),
        "simple_f5_param_mae": float(ff["param_mae"]),
        "simple_late_param_norm_mae": float(lf["param_norm_mae"]),
        "simple_late_param_corr": float(lf["param_corr"]),
        "simple_late_param_mae": float(lf["param_mae"]),
        "simple_sum_param_norm": float(ff["param_norm_mae"] + lf["param_norm_mae"]),
    }


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    dev = coeff.make_dev_rows(tg, st, bp6)
    if len(dev) != 2628 or dev.game_pk.nunique() != 1314:
        raise RuntimeError(f"dev gate {len(dev)}/{dev.game_pk.nunique()}")

    modes = [
        "pooled_calendar30_raw",
        "equal_team_calendar30",
        "equal_day_calendar30",
    ]
    modes += [f"shrink_long_p{int(p)}" for p in PSEUDOS]
    modes += [f"shrink_season_p{int(p)}" for p in PSEUDOS]
    modes += [f"exp30_h{int(h)}" for h in HALFLIVES]

    results = [fit_mode(dev, tg, model, m) for m in modes]
    by_f5 = sorted(results, key=lambda x: x["simple_f5_param_norm_mae"])
    by_late = sorted(results, key=lambda x: x["simple_late_param_norm_mae"])
    by_sum = sorted(results, key=lambda x: x["simple_sum_param_norm"])

    payload = {
        "version": "mu-v1.2-30d-smoothing-semantics-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "historical_constraint": "Only pregame calendar-30D-compatible aggregations are tested; no arbitrary N-game window is eligible here.",
        "best_f5": by_f5[0],
        "best_late": by_late[0],
        "best_combined": by_sum[0],
        "raw_calendar30": next(x for x in results if x["mode"] == "pooled_calendar30_raw"),
        "top10_f5": by_f5[:10],
        "top10_late": by_late[:10],
        "top10_combined": by_sum[:10],
        "all": results,
        "decision_rule": "Frozen 2024-2025 simple coefficient parity is primary. A smoother may replace raw calendar30 only if it materially improves parity while remaining consistent with the preserved 30D/no-same-day-leakage documentation.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

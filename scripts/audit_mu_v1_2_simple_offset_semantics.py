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
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_simple_offset_semantics.json"


def _slice_games(hist: pd.DataFrame, n_games: int) -> pd.DataFrame:
    ids = (
        hist[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"])
        .tail(int(n_games))["game_pk"]
        .astype(str)
        .tolist()
    )
    return hist[hist.game_pk.astype(str).isin(ids)]


def league_offsets(tg: pd.DataFrame, dates: list[pd.Timestamp], mode: str) -> dict[pd.Timestamp, tuple[float, float]]:
    out: dict[pd.Timestamp, tuple[float, float]] = {}
    for d0 in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
        d = pd.Timestamp(d0).normalize()
        if mode == "calendar30_strict":
            h = tg[(tg.game_date < d) & (tg.game_date >= d - pd.Timedelta(days=30))]
        elif mode == "calendar30_exclusive_lower":
            h = tg[(tg.game_date < d) & (tg.game_date > d - pd.Timedelta(days=30))]
        elif mode == "season_calendar30_strict":
            h = tg[(tg.game_date < d) & (tg.game_date >= d - pd.Timedelta(days=30)) & (tg.game_date.dt.year == d.year)]
        elif mode == "last30_league_games":
            pre = tg[tg.game_date < d]
            h = _slice_games(pre, 30)
        elif mode == "last150_league_games":
            pre = tg[tg.game_date < d]
            h = _slice_games(pre, 150)
        elif mode == "calendar30_same_day":
            h = tg[(tg.game_date <= d) & (tg.game_date >= d - pd.Timedelta(days=30))]
        else:
            raise ValueError(mode)
        if h.empty:
            pre = tg[tg.game_date < d]
            h = pre.tail(min(len(pre), 300))
        f5 = float(h.runs_f5.mean())
        if "late_6plus_pa" not in h.columns:
            raise RuntimeError("late_6plus_pa missing")
        late = float(h.late_6plus_pa.mean())
        out[d] = (f5, late)
    return out


def apply_offsets(rows: pd.DataFrame, tg: pd.DataFrame, mode: str) -> pd.DataFrame:
    z = rows.copy()
    dates = pd.to_datetime(z.game_date).dt.normalize().tolist()
    mp = league_offsets(tg, dates, mode)
    f5, late = [], []
    for d0 in pd.to_datetime(z.game_date).dt.normalize():
        a, b = mp[pd.Timestamp(d0)]
        f5.append(a); late.append(b)
    z["league_f5_30d"] = np.asarray(f5, float)
    z["league_late_30d"] = np.asarray(late, float)
    return z


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

    modes = [
        "calendar30_strict",
        "calendar30_exclusive_lower",
        "season_calendar30_strict",
        "last30_league_games",
        "last150_league_games",
        "calendar30_same_day",
    ]
    pub_f5 = float(metrics["oos_2026"]["f5"]["hybrid"]["mean_mu"])
    pub_full = float(metrics["oos_2026"]["full"]["hybrid"]["mean_mu"])
    results = []
    for mode in modes:
        d = apply_offsets(dev, tg, mode)
        # Historical late target forensic winner is inning-6-plus PA runs.
        f5fit = coeff.fit_component(d, model, "simple_f5", engine.SIMPLE_F5, "runs_f5", "league_f5_30d", 1.0)
        latefit = coeff.fit_component(d, model, "simple_late", engine.SIMPLE_LATE, "runs_late_6plus", "league_late_30d", 1.0)
        oo = apply_offsets(oos, tg, mode)
        mm = mean_mu(oo, model)
        results.append({
            "mode": mode,
            "simple_f5_param_norm_mae": float(f5fit["param_norm_mae"]),
            "simple_f5_param_corr": float(f5fit["param_corr"]),
            "simple_late_param_norm_mae": float(latefit["param_norm_mae"]),
            "simple_late_param_corr": float(latefit["param_corr"]),
            "simple_sum_param_norm": float(f5fit["param_norm_mae"] + latefit["param_norm_mae"]),
            "oos_mean_mu_f5": mm["f5"],
            "oos_mean_mu_full": mm["full"],
            "delta_published_f5_mean_mu": float(mm["f5"] - pub_f5),
            "delta_published_full_mean_mu": float(mm["full"] - pub_full),
            "oos_total_gap_abs_sum": float(abs(mm["f5"] - pub_f5) + abs(mm["full"] - pub_full)),
        })
    by_coeff = sorted(results, key=lambda x: x["simple_sum_param_norm"])
    by_oos = sorted(results, key=lambda x: x["oos_total_gap_abs_sum"])
    payload = {
        "version": "mu-v1.2-simple-offset-semantics-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(dev)),
        "development_games": int(dev.game_pk.nunique()),
        "oos_rows": int(len(oos)),
        "oos_games": int(oos.game_pk.nunique()),
        "published_mean_mu": {"f5": pub_f5, "full": pub_full},
        "ranked_by_frozen_simple_coefficient_parity": by_coeff,
        "ranked_by_2026_mean_mu_match_posthoc_only": by_oos,
        "decision_rule": "Use 2024-2025 frozen simple coefficient parity as historical evidence. 2026 mean-mu match is secondary/posthoc and must not choose the historical formula by itself.",
        "note": "Exact original feature-row parity cannot be tested because original v1.2 commits preserve only README/model_spec/metrics, not row-level feature or prediction dumps.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

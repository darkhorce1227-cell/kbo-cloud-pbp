from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_core_side_ablation.json"

RATE = ["sp_kbb_long", "sp_kbb_recent5", "sp_hr_long", "sp_hr_recent5", "sp_h_long", "sp_h_recent5"]
IP = ["sp_ip_long", "sp_ip_recent5"]
STARTER = RATE + IP
BP = ["opp_bp_ra9_long_rel", "opp_bp_ra9_30d_rel"]


def swap_within_game(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = rows.copy(deep=True)
    for _, idx in z.groupby("game_pk").groups.items():
        ids = list(idx)
        if len(ids) != 2:
            continue
        a, b = ids
        av = z.loc[a, cols].copy()
        bv = z.loc[b, cols].copy()
        z.loc[a, cols] = bv.to_numpy()
        z.loc[b, cols] = av.to_numpy()
    return z


def neutral(rows: pd.DataFrame, cols: list[str], model: dict) -> pd.DataFrame:
    z = rows.copy(deep=True)
    means = model["components"]["core_f5"]["scaler_mean"]
    for c in cols:
        if c in means:
            z[c] = float(means[c])
    return z


def summarize_fast(rows: pd.DataFrame, model: dict) -> dict:
    acc = {
        "f5": {"abs": [], "hit": [], "pred_margin": [], "actual_margin": []},
        "full": {"abs": [], "hit": [], "pred_margin": [], "actual_margin": []},
    }
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away = x[x.is_home == 0].iloc[0].to_dict()
        home = x[x.is_home == 1].iloc[0].to_dict()
        pred = engine.score(model, away, home)
        for kind, actual_col, ka, kh in [
            ("f5", "runs_f5", "f5_away", "f5_home"),
            ("full", "runs_full", "full_away", "full_home"),
        ]:
            ya, yh = float(away[actual_col]), float(home[actual_col])
            ma, mh = float(pred[ka]), float(pred[kh])
            acc[kind]["abs"].extend([abs(ya - ma), abs(yh - mh)])
            acc[kind]["pred_margin"].append(ma - mh)
            acc[kind]["actual_margin"].append(ya - yh)
            if ya != yh:
                acc[kind]["hit"].append(1.0 if ((ma > mh) == (ya > yh)) else 0.0)
    out = {}
    for kind, v in acc.items():
        pm = np.asarray(v["pred_margin"], dtype=float)
        am = np.asarray(v["actual_margin"], dtype=float)
        corr = float(np.corrcoef(pm, am)[0, 1]) if len(pm) > 1 and pm.std() > 0 and am.std() > 0 else None
        out[kind] = {
            "mae": float(np.mean(v["abs"])),
            "side": float(np.mean(v["hit"])),
            "margin_corr": corr,
            "decided_games": int(len(v["hit"])),
        }
    return out


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    puball = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    rows = recovered.make_oos_rows(tg, st, bp6)
    if len(rows) != 1368 or rows.game_pk.nunique() != 684:
        raise RuntimeError(f"OOS gate failed rows={len(rows)} games={rows.game_pk.nunique()}")

    variants = {
        "baseline_recovered": rows,
        "swap_starter_rates": swap_within_game(rows, RATE),
        "swap_starter_ip": swap_within_game(rows, IP),
        "swap_starter_all": swap_within_game(rows, STARTER),
        "swap_bullpen": swap_within_game(rows, BP),
        "swap_starter_and_bullpen": swap_within_game(swap_within_game(rows, STARTER), BP),
        "neutral_starter_rates": neutral(rows, RATE, model),
        "neutral_starter_ip": neutral(rows, IP, model),
        "neutral_starter_all": neutral(rows, STARTER, model),
        "neutral_bullpen": neutral(rows, BP, model),
        "neutral_pitching_all": neutral(neutral(rows, STARTER, model), BP, model),
    }

    pub = {
        "f5": {
            "mae": float(puball["oos_2026"]["f5"]["hybrid"]["mae"]),
            "side": float(puball["decided_side_accuracy_2026"]["f5"]),
        },
        "full": {
            "mae": float(puball["oos_2026"]["full"]["hybrid"]["mae"]),
            "side": float(puball["decided_side_accuracy_2026"]["full"]),
        },
    }
    ranked = []
    for name, vr in variants.items():
        s = summarize_fast(vr, model)
        side_distance = abs(s["f5"]["side"] - pub["f5"]["side"]) + abs(s["full"]["side"] - pub["full"]["side"])
        mae_distance = abs(s["f5"]["mae"] - pub["f5"]["mae"]) + abs(s["full"]["mae"] - pub["full"]["mae"])
        ranked.append({
            "variant": name,
            "side_distance": float(side_distance),
            "mae_distance": float(mae_distance),
            "f5": s["f5"],
            "full": s["full"],
        })
    ranked.sort(key=lambda x: (x["side_distance"], x["mae_distance"]))

    payload = {
        "version": "mu-v1.2-core-side-ablation-v0.2-fast",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "rows": int(len(rows)),
        "games": int(rows.game_pk.nunique()),
        "published_target": pub,
        "ranked": ranked,
        "interpretation_rule": "A swap materially reducing side-distance suggests opposite-team mapping for that feature group may match the lost pipeline. Neutral improvement suggests row-level reconstruction error in that group. This is diagnosis only, not promotion permission.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

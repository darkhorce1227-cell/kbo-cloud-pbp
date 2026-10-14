from __future__ import annotations

import json
from pathlib import Path

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


def summarize(rows: pd.DataFrame, model: dict) -> dict:
    e = engine.evaluate(rows, model)
    return {
        "f5": {
            "mae": e["f5"]["team_score_mae"],
            "nb_nll": e["f5"]["nb_nll"],
            "wdl_brier": e["f5"]["wdl_brier_mean3"],
            "wdl_logloss": e["f5"]["wdl_logloss"],
            "side": e["f5"]["decided_side_hit"],
        },
        "full": {
            "mae": e["full"]["team_score_mae"],
            "nb_nll": e["full"]["nb_nll"],
            "wdl_brier": e["full"]["wdl_brier_mean3"],
            "wdl_logloss": e["full"]["wdl_logloss"],
            "side": e["full"]["decided_side_hit"],
        },
    }


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
            **puball["oos_2026"]["f5"]["hybrid"],
            "side": puball["decided_side_accuracy_2026"]["f5"],
        },
        "full": {
            **puball["oos_2026"]["full"]["hybrid"],
            "side": puball["decided_side_accuracy_2026"]["full"],
        },
    }
    results = {}
    for name, vr in variants.items():
        s = summarize(vr, model)
        dist = (
            abs(s["f5"]["side"] - pub["f5"]["side"])
            + abs(s["full"]["side"] - pub["full"]["side"])
            + abs(s["f5"]["wdl_logloss"] - pub["f5"]["wdl_logloss"])
            + abs(s["full"]["wdl_logloss"] - pub["full"]["wdl_logloss"])
        )
        results[name] = {"metrics": s, "distance": float(dist)}

    ranked = sorted(
        [{"variant": k, "distance": v["distance"], **v["metrics"]} for k, v in results.items()],
        key=lambda x: x["distance"],
    )
    payload = {
        "version": "mu-v1.2-core-side-ablation-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "rows": int(len(rows)),
        "games": int(rows.game_pk.nunique()),
        "published_target": pub,
        "ranked": ranked,
        "interpretation_rule": "A swap improving side/logloss materially suggests the original fitted model used the opposite team mapping for that feature group. Neutral improvements instead suggest that group is still row-level misreconstructed. Neither alone authorizes promotion.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

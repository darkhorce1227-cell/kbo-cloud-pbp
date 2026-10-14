from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
PUBLISHED_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_recovered_selfstarter_oos.json"
STARTER = [
    "sp_kbb_long", "sp_kbb_recent5", "sp_hr_long", "sp_hr_recent5",
    "sp_h_long", "sp_h_recent5", "sp_ip_long", "sp_ip_recent5",
]


def swap_starter(rows: pd.DataFrame) -> pd.DataFrame:
    z = rows.copy(deep=True)
    for _, idx in z.groupby("game_pk").groups.items():
        ids = list(idx)
        if len(ids) != 2:
            continue
        a, b = ids
        av = z.loc[a, STARTER].copy()
        bv = z.loc[b, STARTER].copy()
        z.loc[a, STARTER] = bv.to_numpy()
        z.loc[b, STARTER] = av.to_numpy()
    return z


def main():
    tg, st, bp6 = recovered.load_tables()
    rows = recovered.make_oos_rows(tg, st, bp6)
    if len(rows) != 1368 or rows.game_pk.nunique() != 684:
        raise RuntimeError(f"OOS gate failed rows={len(rows)} games={rows.game_pk.nunique()}")
    rows = swap_starter(rows)

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    got = engine.evaluate(rows, model)
    mm = recovered.mean_mu(rows, model)
    pub = published["oos_2026"]
    pub_side = published["decided_side_accuracy_2026"]

    mapped = {}
    delta = {}
    for kind in ("f5", "full"):
        mapped[kind] = {
            "mae": got[kind]["team_score_mae"],
            "nb_nll": got[kind]["nb_nll"],
            "wdl_brier": got[kind]["wdl_brier_mean3"],
            "wdl_logloss": got[kind]["wdl_logloss"],
            "decided_side": got[kind]["decided_side_hit"],
            "mean_mu": mm[kind],
        }
        p = pub[kind]["hybrid"]
        delta[kind] = {
            "mae": mapped[kind]["mae"] - float(p["mae"]),
            "nb_nll": mapped[kind]["nb_nll"] - float(p["nb_nll"]),
            "wdl_brier": mapped[kind]["wdl_brier"] - float(p["wdl_brier"]),
            "wdl_logloss": mapped[kind]["wdl_logloss"] - float(p["wdl_logloss"]),
            "mean_mu": mapped[kind]["mean_mu"] - float(p["mean_mu"]),
            "decided_side": mapped[kind]["decided_side"] - float(pub_side[kind]),
        }

    payload = {
        "version": "mu-v1.2-recovered-selfstarter-oos-v0.2",
        "status": "CHALLENGER_DIAGNOSTIC",
        "official_shadow_allowed": False,
        "oos_team_games": int(len(rows)),
        "oos_games": int(rows.game_pk.nunique()),
        "starter_mapping": "SELF_TEAM_STARTER_BY_WITHIN_GAME_SWAP",
        "bullpen_mapping": "OPPONENT_BULLPEN_UNCHANGED",
        "recovered_2026_oos": mapped,
        "published_v1_2_hybrid_2026_oos": {
            "f5": {**pub["f5"]["hybrid"], "decided_side": pub_side["f5"]},
            "full": {**pub["full"]["hybrid"], "decided_side": pub_side["full"]},
        },
        "delta_recovered_minus_published": delta,
        "note": "Diagnostic only. Frozen v1.2 model coefficients/scalers/gamma/NB alpha are unchanged; only starter-team mapping is swapped relative to v0.1.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

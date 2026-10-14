from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_row_semantics.json"

OFF_F5 = ["off_f5_long_ratio", "off_f5_30d_ratio"]


def swap_cols_within_game(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def score_variant(rows: pd.DataFrame, frozen: dict, name: str) -> dict:
    simple = coeff.fit_component(
        rows, frozen, "simple_f5", engine.SIMPLE_F5,
        "runs_f5", "league_f5_30d", 1.0,
    )
    core = coeff.fit_component(
        rows, frozen, "core_f5", engine.CORE_F5,
        "runs_f5", "league_f5_30d", 5.0,
    )
    return {
        "variant": name,
        "score": float(simple["param_norm_mae"] + core["param_norm_mae"]),
        "simple_f5": {
            "param_norm_mae": simple["param_norm_mae"],
            "param_corr": simple["param_corr"],
            "param_mae": simple["param_mae"],
        },
        "core_f5": {
            "param_norm_mae": core["param_norm_mae"],
            "param_corr": core["param_corr"],
            "param_mae": core["param_mae"],
        },
    }


def main():
    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    base = coeff.make_dev_rows(tg, st, bp6)
    if len(base) != 2628 or base.game_pk.nunique() != 1314:
        raise RuntimeError("development gate failed")

    self_starter = lateaudit.swap_starter(base)
    offense_home = swap_cols_within_game(base, OFF_F5 + ["is_home"])
    self_plus_offense_home = swap_cols_within_game(self_starter, OFF_F5 + ["is_home"])

    variants = [
        score_variant(base, frozen, "OPP_STARTER_CURRENT_ROW"),
        score_variant(self_starter, frozen, "SELF_STARTER_ONLY"),
        score_variant(offense_home, frozen, "SWAP_OFFENSE_HOME_ONLY"),
        score_variant(self_plus_offense_home, frozen, "SELF_STARTER_PLUS_SWAP_OFFENSE_HOME"),
    ]
    variants.sort(key=lambda x: x["score"])

    payload = {
        "version": "mu-v1.2-row-semantics-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(base)),
        "development_games": int(base.game_pk.nunique()),
        "ranked": variants,
        "interpretation_rule": (
            "If SELF_STARTER_ONLY wins while swapping offense/home worsens parity, "
            "the evidence favors a historical starter-mapping implementation bug over a wholesale team-row inversion."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

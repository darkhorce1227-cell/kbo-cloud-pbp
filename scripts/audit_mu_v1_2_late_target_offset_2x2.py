from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset
import audit_mu_v1_2_recovered_oos as recovered

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_late_target_offset_2x2.json"


def beta_vector(result: dict, frozen_comp: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = ["const"] + list(frozen_comp["features"])
    fit = np.array([float(result["params_fit"][k]) for k in keys], dtype=float)
    ref = np.array([float(result["params_frozen"][k]) for k in keys], dtype=float)
    return fit, ref


def fit_combo(base: pd.DataFrame, tg: pd.DataFrame, frozen: dict, target_kind: str, offset_kind: str) -> dict:
    rows = lateoffset.apply_offset_only(base, tg, "late_6plus_pa" if offset_kind == "6plus" else "late_6to9_pa")
    rows = lateaudit.swap_starter(rows)
    target_col = "runs_late_6plus" if target_kind == "6plus" else "runs_late_6to9"
    simple = coeff.fit_component(rows, frozen, "simple_late", engine.SIMPLE_LATE, target_col, "league_late_30d", 1.0)
    core = coeff.fit_component(rows, frozen, "core_late", engine.CORE_LATE, target_col, "league_late_30d", 300.0)
    return {
        "target": target_kind,
        "offset": offset_kind,
        "score": float(simple["param_norm_mae"] + core["param_norm_mae"]),
        "simple_late": {
            "param_norm_mae": float(simple["param_norm_mae"]),
            "param_mae": float(simple["param_mae"]),
            "param_corr": float(simple["param_corr"]),
            "params_fit": simple["params_fit"],
            "params_frozen": simple["params_frozen"],
        },
        "core_late": {
            "param_norm_mae": float(core["param_norm_mae"]),
            "param_mae": float(core["param_mae"]),
            "param_corr": float(core["param_corr"]),
            "params_fit": core["params_fit"],
            "params_frozen": core["params_frozen"],
        },
    }


def projection(ref: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    d = b1 - b0
    den = float(np.dot(d, d))
    return float(np.dot(ref - b0, d) / den) if den > 0 else float("nan")


def main() -> None:
    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    base = coeff.make_dev_rows(tg, st, bp6)

    combos = []
    for target in ("6plus", "6to9"):
        for offset in ("6plus", "6to9"):
            combos.append(fit_combo(base, tg, frozen, target, offset))
    combos.sort(key=lambda x: x["score"])

    # Geometry: with offset held fixed, where does frozen beta lie on the 6to9 -> 6plus target direction?
    proj = {}
    for offset in ("6plus", "6to9"):
        c9 = next(x for x in combos if x["target"] == "6to9" and x["offset"] == offset)
        c6 = next(x for x in combos if x["target"] == "6plus" and x["offset"] == offset)
        for comp_name in ("simple_late", "core_late"):
            keys = ["const"] + list(frozen["components"][comp_name]["features"])
            b9 = np.array([float(c9[comp_name]["params_fit"][k]) for k in keys])
            b6 = np.array([float(c6[comp_name]["params_fit"][k]) for k in keys])
            ref = np.array([float(c6[comp_name]["params_frozen"][k]) for k in keys])
            proj[f"offset_{offset}_{comp_name}"] = projection(ref, b9, b6)

    extra = base["runs_late_6plus"] - base["runs_late_6to9"]
    extra_games = base.loc[extra > 0, "game_pk"].nunique()
    extra_rows = int((extra > 0).sum())

    payload = {
        "version": "mu-v1.2-late-target-offset-2x2-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(base)),
        "development_games": int(base.game_pk.nunique()),
        "extra_inning_signal": {
            "rows_with_runs_after_9th": extra_rows,
            "games_with_runs_after_9th": int(extra_games),
            "total_runs_after_9th": float(extra.clip(lower=0).sum()),
            "mean_runs_after_9th_per_team_game": float(extra.clip(lower=0).mean()),
        },
        "ranked_2x2": combos,
        "target_direction_projection": proj,
        "projection_interpretation": "0 means frozen beta lies at 6to9 fit along the target-semantic direction; 1 means at 6plus fit; values outside [0,1] mean the frozen vector lies beyond that segment.",
        "decision_rule": "Historical semantics should be selected primarily by frozen coefficient parity, not 2026 aggregate OOS fit. Require target and offset evidence to be directionally coherent before changing the Challenger default.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

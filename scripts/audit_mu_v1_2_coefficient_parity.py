from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_coefficient_parity.json"
TRAIN_START = pd.Timestamp("2024-05-01")


def make_dev_rows(tg, st, bp6):
    target = tg[
        ((tg.game_date.dt.year == 2024) & (tg.game_date >= TRAIN_START))
        | (tg.game_date.dt.year == 2025)
    ].sort_values(["game_date", "game_pk", "is_home"])
    rows = []
    for r in target.itertuples(index=False):
        f = recovered.build_features(tg, st, bp6, row=r)
        f.update({
            "game_pk": str(r.game_pk),
            "game_date": pd.Timestamp(r.game_date).normalize(),
            "team": str(r.team),
            "opponent": str(r.opponent),
            "is_home": float(r.is_home),
            "runs_f5": float(r.runs_f5),
            "runs_late": float(r.runs_late),
            "runs_late_6plus": float(r.late_6plus_pa),
            "runs_late_6to9": float(r.late_6to9_pa),
            "runs_full": float(r.runs_full),
        })
        rows.append(f)
    df = pd.DataFrame(rows)
    if len(df) != 2628 or df.game_pk.nunique() != 1314:
        raise RuntimeError(f"development gate failed rows={len(df)} games={df.game_pk.nunique()}")
    return df


def comp_distance(fit: dict, frozen: dict) -> dict:
    feats = frozen["features"]
    keys = ["const"] + feats
    p_fit = np.array([float(fit["params"][k]) for k in keys])
    p_ref = np.array([float(frozen["params"][k]) for k in keys])
    s_fit = np.array([float(fit["scaler_scale"][k]) for k in feats])
    s_ref = np.array([float(frozen["scaler_scale"][k]) for k in feats])
    m_fit = np.array([float(fit["scaler_mean"][k]) for k in feats])
    m_ref = np.array([float(frozen["scaler_mean"][k]) for k in feats])
    param_abs = np.abs(p_fit - p_ref)
    param_norm = param_abs / np.maximum(np.abs(p_ref), 0.005)
    return {
        "param_mae": float(param_abs.mean()),
        "param_max_abs": float(param_abs.max()),
        "param_norm_mae": float(param_norm.mean()),
        "param_corr": float(np.corrcoef(p_fit, p_ref)[0, 1]) if len(p_fit) > 1 else 1.0,
        "scaler_mean_mae": float(np.abs(m_fit - m_ref).mean()),
        "scaler_scale_mae": float(np.abs(s_fit - s_ref).mean()),
        "params_fit": {k: float(fit["params"][k]) for k in keys},
        "params_frozen": {k: float(frozen["params"][k]) for k in keys},
    }


def fit_component(rows, frozen_model, comp, features, target, offset, alpha):
    effective_alpha = float(alpha) * float(len(rows))
    fit = engine.fit_poisson_ridge(rows, list(features), target, offset, effective_alpha)
    d = comp_distance(fit, frozen_model["components"][comp])
    d["declared_alpha"] = float(alpha)
    d["effective_alpha"] = effective_alpha
    d["target_used"] = target
    d["offset_used"] = offset
    return d


def fit_variant(rows: pd.DataFrame, frozen_model: dict, late_target: str, label: str) -> dict:
    distances = {
        "simple_f5": fit_component(rows, frozen_model, "simple_f5", engine.SIMPLE_F5, "runs_f5", "league_f5_30d", 1.0),
        "core_f5": fit_component(rows, frozen_model, "core_f5", engine.CORE_F5, "runs_f5", "league_f5_30d", 5.0),
        "simple_late": fit_component(rows, frozen_model, "simple_late", engine.SIMPLE_LATE, late_target, "league_late_30d", 1.0),
        "core_late": fit_component(rows, frozen_model, "core_late", engine.CORE_LATE, late_target, "league_late_30d", 300.0),
    }
    return {
        "variant": label,
        "late_target": late_target,
        "late_param_norm_score": float(distances["simple_late"]["param_norm_mae"] + distances["core_late"]["param_norm_mae"]),
        "all_param_norm_score": float(sum(v["param_norm_mae"] for v in distances.values())),
        "components": distances,
    }


def main():
    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg, st, bp6 = recovered.load_tables()
    base = make_dev_rows(tg, st, bp6)

    definitions = [
        ("FULL_MINUS_F5", base, "runs_late"),
        ("PA_INNING6PLUS", lateoffset.apply_offset_only(base, tg, "late_6plus_pa"), "runs_late_6plus"),
        ("PA_INNING6TO9", lateoffset.apply_offset_only(base, tg, "late_6to9_pa"), "runs_late_6to9"),
        # Crossed definitions retained only to diagnose historical train/offset mismatch.
        ("TARGET_FULLMINUS_OFFSET_6PLUS", lateoffset.apply_offset_only(base, tg, "late_6plus_pa"), "runs_late"),
        ("TARGET_FULLMINUS_OFFSET_6TO9", lateoffset.apply_offset_only(base, tg, "late_6to9_pa"), "runs_late"),
    ]

    results = []
    for label, opp_rows, target_col in definitions:
        self_rows = lateaudit.swap_starter(opp_rows)
        results.append(fit_variant(self_rows, frozen, target_col, label))
    results.sort(key=lambda x: x["all_param_norm_score"])

    payload = {
        "version": "mu-v1.2-coefficient-parity-v0.4",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "development_rows": int(len(base)),
        "development_games": int(base.game_pk.nunique()),
        "fixed_recovered_semantics": {
            "starter_mapping": "SELF_STARTER",
            "late_offense_features": "FULL_MINUS_F5",
            "bullpen": "INNING6PLUS_SEASON_RESET_CALENDAR30_P360_P54_OPPONENT",
            "starter_rate_prior": "league->long 150 BF; long->recent5 150 BF",
            "starter_ip_prior": "league->long 5 starts; long->recent5 5 starts",
            "penalty_mode": "MEAN_LOSS_ALPHA",
        },
        "ranked_late_target_offset_definitions": results,
        "interpretation": "Use frozen coefficient parity to identify the original training target/offset semantics. Do not select a definition merely because it reproduces 2026 OOS aggregate metrics better.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from reverse_mu_v1_2_preprocessing import (
    SPEC, FROZEN, TRAIN_START, build_raw_rows, offense_features,
    sp_rate_features, sp_ip_features, bp_features, park_features,
)
from run_mu_v1_2_pipeline import prepare_pa_fixed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "state" / "mu_v1_2_candidate_preprocessing_audit.json"

CANDIDATE = {
    "offense_pseudo_games_long": 25.0,
    "offense_pseudo_games_30d": 5.0,
    "starter_rate_pseudo_pa_long": 125.0,
    "starter_rate_pseudo_pa_recent5": 60.0,
    "starter_ip_pseudo_starts_long": 4.0,
    "starter_ip_pseudo_starts_recent5": 1.5,
    "bullpen_pseudo_outs_long": 0.0,
    "bullpen_pseudo_outs_30d": 0.0,
    "park_pseudo_games": 5.0,
}


def feature_moment(vals: np.ndarray, feat: str) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    m = float(vals.mean())
    s = float(vals.std(ddof=0))
    fm = float(FROZEN[feat]["mean"])
    fs = float(FROZEN[feat]["scale"])
    return {
        "n": int(len(vals)),
        "candidate_mean": m,
        "frozen_mean": fm,
        "abs_mean_diff": abs(m - fm),
        "candidate_scale": s,
        "frozen_scale": fs,
        "abs_scale_diff": abs(s - fs),
        "z_mean_abs_error": abs(m - fm) / max(abs(fs), 1e-12),
        "relative_scale_error": abs(s - fs) / max(abs(fs), 1e-12),
    }


def main() -> None:
    df = mu.load_seasons(2024)
    pa = prepare_pa_fixed(df)
    tg, st, bp = mu.game_tables(pa)
    raw = build_raw_rows(tg, st, bp)

    feats: dict[str, np.ndarray] = {}
    feats.update(offense_features(raw, CANDIDATE["offense_pseudo_games_long"], CANDIDATE["offense_pseudo_games_30d"]))
    feats.update(sp_rate_features(raw, CANDIDATE["starter_rate_pseudo_pa_long"], CANDIDATE["starter_rate_pseudo_pa_recent5"]))
    feats.update(sp_ip_features(raw, CANDIDATE["starter_ip_pseudo_starts_long"], CANDIDATE["starter_ip_pseudo_starts_recent5"]))
    feats.update(bp_features(raw, CANDIDATE["bullpen_pseudo_outs_long"], CANDIDATE["bullpen_pseudo_outs_30d"]))
    feats["park_factor"] = park_features(raw, CANDIDATE["park_pseudo_games"])
    feats["is_home"] = raw["is_home"].to_numpy(float)

    checks = {k: feature_moment(v, k) for k, v in feats.items()}
    max_mean = max(v["abs_mean_diff"] for v in checks.values())
    max_scale = max(v["abs_scale_diff"] for v in checks.values())
    worst = sorted(
        checks.items(),
        key=lambda kv: kv[1]["z_mean_abs_error"] + kv[1]["relative_scale_error"],
        reverse=True,
    )

    payload = {
        "version": "mu-v1.2-inferred-preprocessing-candidate-v0.1",
        "status": "CANDIDATE_NOT_PARITY",
        "official_shadow_allowed": False,
        "training_rows": int(len(raw)),
        "unique_games": int(raw["game_pk"].nunique()),
        "candidate": CANDIDATE,
        "max_abs_mean_diff": float(max_mean),
        "max_abs_scale_diff": float(max_scale),
        "worst_features": [
            {
                "feature": k,
                "combined_standardized_error": float(v["z_mean_abs_error"] + v["relative_scale_error"]),
                **v,
            }
            for k, v in worst[:8]
        ],
        "checks": checks,
        "interpretation": {
            "close_groups": ["offense ratios", "starter IP", "park factor", "home flag"],
            "unresolved_groups": ["starter K-BB/HR/H rates", "bullpen RA9 relative"],
            "warning": "Candidate priors were reverse-engineered from frozen moments. Do not promote or generate official Shadow until the unresolved feature definitions are reproduced without target-fitting shortcuts.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

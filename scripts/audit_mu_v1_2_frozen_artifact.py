from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import (
    FROZEN, build_raw_rows, offense_features, sp_rate_features,
    sp_ip_features, bp_features, park_features,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_2024_SHA256 = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
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


def moment(vals, feat):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    m, s = float(a.mean()), float(a.std(ddof=0))
    fm, fs = float(FROZEN[feat]["mean"]), float(FROZEN[feat]["scale"])
    return {
        "n": int(len(a)), "candidate_mean": m, "frozen_mean": fm,
        "abs_mean_diff": abs(m-fm), "candidate_scale": s, "frozen_scale": fs,
        "abs_scale_diff": abs(s-fs),
        "standardized_error": abs(m-fm)/max(abs(fs),1e-12) + abs(s-fs)/max(abs(fs),1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", default="out_training")
    ap.add_argument("--out", default="state/mu_v1_2_frozen_artifact_audit.json")
    args = ap.parse_args()
    training = ROOT / args.training_dir
    p24 = training / "kbo_pbp_2024_training.csv.gz"
    if not p24.exists():
        raise FileNotFoundError(p24)
    sha = hashlib.sha256(p24.read_bytes()).hexdigest()
    if sha != EXPECTED_2024_SHA256:
        raise RuntimeError(f"frozen 2024 artifact SHA mismatch: {sha}")

    df = pd.read_csv(p24, low_memory=False)
    if len(df) != 227955 or df["game_pk"].nunique() != 736:
        raise RuntimeError(f"unexpected frozen rows/games: {len(df)}/{df['game_pk'].nunique()}")
    pa = prepare_pa_fixed(df)
    tg, st, bp = mu.game_tables(pa)
    raw = build_raw_rows(tg, st, bp)

    feats = {}
    feats.update(offense_features(raw, 25.0, 5.0))
    feats.update(sp_rate_features(raw, 125.0, 60.0))
    feats.update(sp_ip_features(raw, 4.0, 1.5))
    feats.update(bp_features(raw, 0.0, 0.0))
    feats["park_factor"] = park_features(raw, 5.0)
    feats["is_home"] = raw["is_home"].to_numpy(float)

    checks = {k: moment(v, k) for k, v in feats.items()}
    worst = sorted(checks.items(), key=lambda kv: kv[1]["standardized_error"], reverse=True)
    payload = {
        "version": "mu-v1.2-frozen-artifact-audit-v0.1",
        "status": "FROZEN_ARTIFACT_DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "artifact": {"sha256": sha, "rows": int(len(df)), "games": int(df.game_pk.nunique())},
        "training_rows": int(len(raw)), "unique_games": int(raw.game_pk.nunique()),
        "candidate": CANDIDATE,
        "max_abs_mean_diff": max(x["abs_mean_diff"] for x in checks.values()),
        "max_abs_scale_diff": max(x["abs_scale_diff"] for x in checks.values()),
        "worst_features": [{"feature": k, **v} for k,v in worst[:10]],
        "checks": checks,
        "interpretation": "If these moments match the fresh-download candidate audit, raw snapshot drift is ruled out and unresolved mismatch is feature-definition/preprocessing logic.",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

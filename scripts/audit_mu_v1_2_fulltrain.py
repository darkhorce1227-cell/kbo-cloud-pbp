from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import build_mu_v1_2_features as mu
import run_mu_v1_2_pipeline as runner

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "research" / "mu-v1.2" / "model_spec.json").read_text(encoding="utf-8"))
OUT = ROOT / "state" / "mechanical_mu_fulltrain_audit.json"
START = pd.Timestamp("2024-05-01")


def main() -> None:
    # Use the corrected PA reconstruction, but otherwise retain the candidate
    # feature definitions exactly as implemented in build_mu_v1_2_features.py.
    mu.prepare_pa = runner.prepare_pa_fixed
    df = mu.load_seasons(2025)
    pa = mu.prepare_pa(df)
    tg, st, bp = mu.game_tables(pa)
    st_map = st.set_index(["game_pk", "pitching_team"])["starter"].to_dict()

    rows = []
    skipped = []
    games = tg[tg["game_date"] >= START].copy()
    for r in games.itertuples(index=False):
        opp_st = st_map.get((r.game_pk, r.opponent))
        if opp_st is None:
            skipped.append({"game_pk": r.game_pk, "team": r.team, "reason": "starter_missing"})
            continue
        try:
            f = mu.build_feature_row(
                tg,
                st,
                bp,
                d=r.game_date,
                team=r.team,
                opponent=r.opponent,
                opposing_starter=opp_st,
                is_home=int(r.is_home),
            )
        except Exception as exc:
            skipped.append({"game_pk": r.game_pk, "team": r.team, "reason": str(exc)})
            continue
        f.update({"game_pk": r.game_pk, "game_date": str(r.game_date.date()), "team": r.team})
        rows.append(f)

    tr = pd.DataFrame(rows)
    feature_names = sorted({k for c in SPEC["components"].values() for k in c["features"]})
    checks = {}
    max_mean = 0.0
    max_scale = 0.0
    for feat in feature_names:
        vals = pd.to_numeric(tr.get(feat), errors="coerce").dropna().to_numpy(float)
        refs = [
            (float(c["scaler_mean"][feat]), float(c["scaler_scale"][feat]))
            for c in SPEC["components"].values()
            if feat in c["features"]
        ]
        ref_mean, ref_scale = refs[0]
        cand_mean = float(vals.mean()) if len(vals) else float("nan")
        cand_scale = float(vals.std(ddof=0)) if len(vals) else float("nan")
        dm = abs(cand_mean - ref_mean)
        ds = abs(cand_scale - ref_scale)
        max_mean = max(max_mean, dm)
        max_scale = max(max_scale, ds)
        checks[feat] = {
            "n": int(len(vals)),
            "candidate_mean": cand_mean,
            "frozen_mean": ref_mean,
            "abs_mean_diff": dm,
            "candidate_scale": cand_scale,
            "frozen_scale": ref_scale,
            "abs_scale_diff": ds,
        }

    payload = {
        "version": "mu-v1.2-fulltrain-forensic-v0.1",
        "documented_train_window": "2024-05-01 through 2025",
        "candidate_rows": int(len(tr)),
        "candidate_games": int(tr["game_pk"].nunique()) if len(tr) else 0,
        "skipped_rows": int(len(skipped)),
        "skip_examples": skipped[:30],
        "status": "EXACT_PARITY_PASS" if max_mean <= 1e-6 and max_scale <= 1e-6 else "UNVERIFIED_PREPROCESSING",
        "max_abs_mean_diff": max_mean,
        "max_abs_scale_diff": max_scale,
        "checks": checks,
        "note": "Forensic audit only. Exact status requires frozen scaler parity; no affine remapping is allowed.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

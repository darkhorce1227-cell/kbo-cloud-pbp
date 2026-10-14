from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
import run_mu_v1_2_pipeline as runner

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "research" / "mu-v1.2" / "model_spec.json").read_text(encoding="utf-8"))
OUT = ROOT / "state" / "mechanical_mu_crossseason_audit.json"
TRAIN_START = pd.Timestamp("2024-05-01")
HISTORY_START = pd.Timestamp("2024-01-01")


def frozen(feat: str):
    for comp in SPEC["components"].values():
        if feat in comp["features"]:
            return float(comp["scaler_mean"][feat]), float(comp["scaler_scale"][feat])
    raise KeyError(feat)


def main() -> None:
    mu.prepare_pa = runner.prepare_pa_fixed
    raw = mu.load_seasons(2025)
    pa = mu.prepare_pa(raw)
    tg, st, bp = mu.game_tables(pa)
    st_map = st.set_index(["game_pk", "pitching_team"])["starter"].to_dict()
    rows = []

    for r in tg[tg.game_date >= TRAIN_START].sort_values(["game_date", "game_pk", "is_home"]).itertuples(index=False):
        d = r.game_date
        pre = tg[(tg.game_date < d) & (tg.game_date >= HISTORY_START)]
        if pre.empty:
            continue
        pre30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
        my = pre[pre.team.astype(str).eq(str(r.team))]
        my30 = pre30[pre30.team.astype(str).eq(str(r.team))]

        lf5 = float(pre.runs_f5.mean())
        ll = float(pre.runs_late.mean())
        lf530 = float(pre30.runs_f5.mean()) if len(pre30) else lf5
        ll30 = float(pre30.runs_late.mean()) if len(pre30) else ll
        off_f5_long = float(my.runs_f5.mean()/lf5) if len(my) and lf5 > 0 else 1.0
        off_late_long = float(my.runs_late.mean()/ll) if len(my) and ll > 0 else 1.0
        off_f5_30 = float(my30.runs_f5.mean()/lf530) if len(my30) and lf530 > 0 else 1.0
        off_late_30 = float(my30.runs_late.mean()/ll30) if len(my30) and ll30 > 0 else 1.0

        sp_pre = st[(st.game_date < d) & (st.game_date >= HISTORY_START)]
        starter = st_map.get((r.game_pk, r.opponent))
        mine = sp_pre[sp_pre.starter.astype(str).eq(str(starter))].sort_values("game_date") if starter is not None else sp_pre.iloc[0:0]
        mine5 = mine.tail(5)

        def league_rates():
            pa_n = float(sp_pre.pa.sum())
            if pa_n <= 0:
                return 0.0, 0.0, 0.0, 5.0
            return (
                float((sp_pre.k.sum()-sp_pre.bb.sum())/pa_n),
                float(sp_pre.h.sum()/pa_n),
                float(sp_pre.hr.sum()/pa_n),
                float(sp_pre.ip.mean()),
            )
        lkbb, lh, lhr, lip = league_rates()

        def spvals(x):
            if x.empty or x.pa.sum() <= 0:
                return lkbb, lh, lhr, lip
            den = float(x.pa.sum())
            return (
                float((x.k.sum()-x.bb.sum())/den),
                float(x.h.sum()/den),
                float(x.hr.sum()/den),
                float(x.ip.mean()),
            )
        kbb, h, hr, ip = spvals(mine)
        kbb5, h5, hr5, ip5 = spvals(mine5)

        bp_pre = bp[(bp.game_date < d) & (bp.game_date >= HISTORY_START)]
        bp30 = bp_pre[bp_pre.game_date >= d - pd.Timedelta(days=30)]
        def ra9(x):
            outs = float(x.bp_outs.sum())
            return float(x.bp_runs.sum()*27/outs) if outs > 0 else np.nan
        lbp = ra9(bp_pre)
        lbp30 = ra9(bp30) if len(bp30) else lbp
        obp = bp_pre[bp_pre.pitching_team.astype(str).eq(str(r.opponent))]
        obp30 = bp30[bp30.pitching_team.astype(str).eq(str(r.opponent))]
        bp_long_rel = ra9(obp)/lbp if len(obp) and np.isfinite(lbp) and lbp > 0 else 1.0
        bp30_rel = ra9(obp30)/lbp30 if len(obp30) and np.isfinite(lbp30) and lbp30 > 0 else 1.0

        home_team = r.team if int(r.is_home) else r.opponent
        park_games = pre[(pre.team.astype(str).eq(str(home_team))) & (pre.is_home == 1)]
        if len(park_games):
            ids = park_games.game_pk.unique()
            pg = pre[pre.game_pk.isin(ids)]
            league_full = float(pre.runs_full.mean())
            n_games = float(len(ids))
            park_rate = (float(pg.runs_full.sum()) + 40.0*league_full)/(2.0*(n_games+20.0))
            park = park_rate/league_full if league_full > 0 else 1.0
        else:
            park = 1.0

        rows.append({
            "is_home": float(r.is_home),
            "off_f5_long_ratio": off_f5_long, "off_f5_30d_ratio": off_f5_30,
            "off_late_long_ratio": off_late_long, "off_late_30d_ratio": off_late_30,
            "sp_kbb_long": kbb, "sp_kbb_recent5": kbb5,
            "sp_h_long": h, "sp_h_recent5": h5,
            "sp_hr_long": hr, "sp_hr_recent5": hr5,
            "sp_ip_long": ip, "sp_ip_recent5": ip5,
            "opp_bp_ra9_long_rel": bp_long_rel, "opp_bp_ra9_30d_rel": bp30_rel,
            "park_factor": park,
        })

    df = pd.DataFrame(rows)
    checks = {}
    max_mean = max_scale = 0.0
    for feat in df.columns:
        vals = pd.to_numeric(df[feat], errors="coerce").dropna().to_numpy(float)
        fm, fs = frozen(feat)
        m, s = float(vals.mean()), float(vals.std(ddof=0))
        dm, ds = abs(m-fm), abs(s-fs)
        max_mean, max_scale = max(max_mean, dm), max(max_scale, ds)
        checks[feat] = {"n": len(vals), "candidate_mean": m, "frozen_mean": fm, "abs_mean_diff": dm,
                        "candidate_scale": s, "frozen_scale": fs, "abs_scale_diff": ds}
    payload = {
        "version":"mu-v1.2-crossseason-forensic-v0.1",
        "rows":len(df),
        "status":"EXACT_PARITY_PASS" if max_mean <= 1e-6 and max_scale <= 1e-6 else "UNVERIFIED_PREPROCESSING",
        "max_abs_mean_diff":max_mean,"max_abs_scale_diff":max_scale,"checks":checks,
        "note":"Cross-season history hypothesis only; no extra shrink added."
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT = Path(__file__).resolve().parents[1]
SHA24 = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25 = "017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"
P_LONG = 360.0
P_RECENT = 54.0


def dev(tg: pd.DataFrame) -> pd.DataFrame:
    d = tg.game_date
    return tg[((d.dt.year == 2024) & (d >= TRAIN_START)) | (d.dt.year == 2025)].sort_values(
        ["game_date", "game_pk", "is_home"]
    ).reset_index(drop=True)


def moment(vals: np.ndarray, feat: str) -> dict:
    vals = np.asarray(vals, float)
    fm = float(FROZEN[feat]["mean"])
    fs = float(FROZEN[feat]["scale"])
    m = float(np.nanmean(vals))
    s = float(np.nanstd(vals, ddof=0))
    return {
        "mean": m,
        "scale": s,
        "frozen_mean": fm,
        "frozen_scale": fs,
        "mean_error": m - fm,
        "scale_error": s - fs,
        "loss": abs(m - fm) / fs + abs(s - fs) / fs,
    }


def safe_ra9(runs, outs, fallback):
    return float(runs * 27.0 / outs) if outs > 0 else float(fallback)


def build_rows(tg: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in dev(tg).itertuples(index=False):
        d = pd.Timestamp(r.game_date)
        opp = str(r.opponent)

        pre_all = bp[bp.game_date < d]
        pre = pre_all[pre_all.game_date.dt.year == d.year]
        if pre.empty:
            pre = pre_all
        opp_long = pre[pre.pitching_team.astype(str).eq(opp)]

        recent = bp[(bp.game_date < d) & (bp.game_date >= d - pd.Timedelta(days=30))]
        opp_recent = recent[recent.pitching_team.astype(str).eq(opp)]

        lg_long = safe_ra9(float(pre.runs.sum()), float(pre.outs.sum()), 4.5)
        lg_recent = safe_ra9(float(recent.runs.sum()), float(recent.outs.sum()), lg_long)

        # leave-opponent-out league recent / long baselines
        pre_loo = pre[~pre.pitching_team.astype(str).eq(opp)]
        rec_loo = recent[~recent.pitching_team.astype(str).eq(opp)]
        lg_long_loo = safe_ra9(float(pre_loo.runs.sum()), float(pre_loo.outs.sum()), lg_long)
        lg_recent_loo = safe_ra9(float(rec_loo.runs.sum()), float(rec_loo.outs.sum()), lg_recent)

        tr = float(opp_long.runs.sum()); to = float(opp_long.outs.sum())
        rr = float(opp_recent.runs.sum()); ro = float(opp_recent.outs.sum())

        team_long = (tr + P_LONG * lg_long / 27.0) * 27.0 / (to + P_LONG) if to + P_LONG > 0 else lg_long
        long_rel = team_long / lg_long if lg_long else 1.0

        team_recent_to_long = (
            (rr + P_RECENT * team_long / 27.0) * 27.0 / (ro + P_RECENT)
            if ro + P_RECENT > 0 else team_long
        )
        team_recent_to_lg_recent = (
            (rr + P_RECENT * lg_recent / 27.0) * 27.0 / (ro + P_RECENT)
            if ro + P_RECENT > 0 else lg_recent
        )
        team_recent_to_lg_long = (
            (rr + P_RECENT * lg_long / 27.0) * 27.0 / (ro + P_RECENT)
            if ro + P_RECENT > 0 else lg_long
        )

        # Shrink the raw relative ratio itself toward long_rel using outs as weight.
        raw_recent_ra9 = safe_ra9(rr, ro, team_long)
        raw_recent_rel = raw_recent_ra9 / lg_recent if lg_recent else 1.0
        ratio_shrunk = (ro * raw_recent_rel + P_RECENT * long_rel) / (ro + P_RECENT) if ro + P_RECENT > 0 else long_rel

        out.append({
            "game_pk": str(r.game_pk),
            "game_date": str(d.date()),
            "team": str(r.team),
            "opponent": opp,
            "long_rel": long_rel,
            # A: current candidate
            "recent_over_league30_prior_teamlong": team_recent_to_long / lg_recent if lg_recent else 1.0,
            # B: recent performance relative to own long quality
            "recent_over_teamlong_prior_teamlong": team_recent_to_long / team_long if team_long else 1.0,
            # C: recent performance relative to season league environment
            "recent_over_leaguelong_prior_teamlong": team_recent_to_long / lg_long if lg_long else 1.0,
            # D: recent shrunk directly to recent league baseline
            "recent_over_league30_prior_league30": team_recent_to_lg_recent / lg_recent if lg_recent else 1.0,
            # E: recent shrunk to season league baseline, divided by recent environment
            "recent_over_league30_prior_leaguelong": team_recent_to_lg_long / lg_recent if lg_recent else 1.0,
            # F/G: leave-opponent-out denominators, keeping current hierarchical prior
            "recent_over_league30_loo_prior_teamlong": team_recent_to_long / lg_recent_loo if lg_recent_loo else 1.0,
            "recent_over_leaguelong_loo_prior_teamlong": team_recent_to_long / lg_long_loo if lg_long_loo else 1.0,
            # H: ratio-space shrink, a common implementation alternative
            "recent_ratio_shrunk": ratio_shrunk,
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", default="out_training")
    ap.add_argument("--out", default="state/mu_v1_2_bp30_relative_semantics.json")
    a = ap.parse_args()

    td = ROOT / a.training_dir
    p24 = td / "kbo_pbp_2024_training.csv.gz"
    p25 = td / "kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest() == SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest() == SHA25

    raw = pd.concat([pd.read_csv(p24, low_memory=False), pd.read_csv(p25, low_memory=False)], ignore_index=True)
    pa = prepare_pa_fixed(raw)
    tg, _, _ = make_team_game(pa)
    bp = bp_game_tables(pa, tg)["inning6plus"]
    rows = build_rows(tg, bp)
    assert len(rows) == 2628

    long_check = moment(rows["long_rel"].to_numpy(float), "opp_bp_ra9_long_rel")
    variants = []
    cols = [c for c in rows.columns if c.startswith("recent_")]
    for c in cols:
        m = moment(rows[c].to_numpy(float), "opp_bp_ra9_30d_rel")
        variants.append({"variant": c, **m})
    variants.sort(key=lambda x: x["loss"])

    payload = {
        "version": "mu-v1.2-bp30-relative-semantics-v0.1",
        "status": "DIAGNOSTIC_ONLY",
        "official_shadow_allowed": False,
        "rows": int(len(rows)),
        "fixed_semantics": {
            "definition": "inning6plus",
            "scope": "season_reset",
            "window": "calendar30",
            "p_long_outs": P_LONG,
            "p_recent_outs": P_RECENT,
        },
        "long_check": long_check,
        "ranked_recent_variants": variants,
        "note": "Only plausible relative/prior semantics are compared. No arbitrary coefficient fitting and no production promotion.",
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"


def merr(vals, feat):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    m = float(a.mean())
    s = float(a.std(ddof=0))
    fm = float(FROZEN[feat]["mean"])
    fs = float(FROZEN[feat]["scale"])
    loss = abs(m - fm) / max(abs(fs), 1e-12) + abs(s - fs) / max(abs(fs), 1e-12)
    return {"mean": m, "scale": s, "frozen_mean": fm, "frozen_scale": fs, "loss": float(loss)}


def score(features):
    detail = {k: merr(v, k) for k, v in features.items()}
    return float(sum(v["loss"] for v in detail.values())), detail


def starter_game_table(pa):
    first = (
        pa.sort_values(["game_date", "game_pk", "inning", "at_bat_number"])
        .groupby(["game_pk", "pitching_team"], as_index=False)
        .first()[["game_pk", "pitching_team", "pitcher", "game_date"]]
        .rename(columns={"pitcher": "starter"})
    )
    p = pa.merge(first[["game_pk", "pitching_team", "starter"]], on=["game_pk", "pitching_team"], how="left")
    p = p[p["pitcher"].astype(str).eq(p["starter"].astype(str))].copy()
    p["pa_all"] = 1
    p["pa_classified"] = p["event"].notna().astype(int)
    p["k"] = p["event"].eq("strikeout").astype(int)
    p["bb"] = p["event"].eq("walk").astype(int)
    p["hbp"] = p["event"].eq("hit_by_pitch").astype(int)
    p["hr"] = p["event"].eq("home_run").astype(int)
    p["h"] = p["event"].isin({"single", "double", "triple", "home_run"}).astype(int)
    return (
        p.groupby(["game_pk", "game_date", "pitching_team", "starter"], as_index=False)
        .agg(
            pa_all=("pa_all", "sum"),
            pa_classified=("pa_classified", "sum"),
            k=("k", "sum"),
            bb=("bb", "sum"),
            hbp=("hbp", "sum"),
            hr=("hr", "sum"),
            h=("h", "sum"),
        )
    )


def target_games(tg):
    return tg[(tg["game_date"] >= TRAIN_START) & (tg["game_date"].dt.year == 2024)].sort_values(
        ["game_date", "game_pk", "is_home"]
    ).reset_index(drop=True)


def precompute_starter_raw(tg, st_alt):
    games = target_games(tg)
    st_map = st_alt.set_index(["game_pk", "pitching_team"])["starter"].to_dict()
    rows = []
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.game_date).normalize()
        ss = pd.Timestamp(year=d.year, month=1, day=1)
        pre = st_alt[(st_alt["game_date"] < d) & (st_alt["game_date"] >= ss)]
        sid = st_map.get((r.game_pk, r.opponent))
        player = pre[pre["starter"].astype(str).eq(str(sid))].sort_values("game_date") if sid is not None else pre.iloc[0:0]
        rec = {"game_pk": r.game_pk}
        for tag, x in [("long", player), ("recent", player.tail(5)), ("lg", pre)]:
            for c in ["pa_all", "pa_classified", "k", "bb", "hbp", "hr", "h"]:
                rec[f"{tag}_{c}"] = float(x[c].sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def starter_features(raw, denom, subtract_hbp, pseudo_long, pseudo_recent):
    out = {k: [] for k in ["sp_kbb_long", "sp_kbb_recent5", "sp_hr_long", "sp_hr_recent5", "sp_h_long", "sp_h_recent5"]}
    lden = raw[f"lg_{denom}"].to_numpy(float)
    lkbb_num = raw["lg_k"].to_numpy(float) - raw["lg_bb"].to_numpy(float)
    if subtract_hbp:
        lkbb_num = lkbb_num - raw["lg_hbp"].to_numpy(float)
    priors = {
        "kbb": np.divide(lkbb_num, lden, out=np.zeros_like(lden), where=lden > 0),
        "hr": np.divide(raw["lg_hr"].to_numpy(float), lden, out=np.zeros_like(lden), where=lden > 0),
        "h": np.divide(raw["lg_h"].to_numpy(float), lden, out=np.zeros_like(lden), where=lden > 0),
    }
    for tag, pseudo, suffix in [("long", float(pseudo_long), "long"), ("recent", float(pseudo_recent), "recent5")]:
        den = raw[f"{tag}_{denom}"].to_numpy(float)
        kbb_num = raw[f"{tag}_k"].to_numpy(float) - raw[f"{tag}_bb"].to_numpy(float)
        if subtract_hbp:
            kbb_num = kbb_num - raw[f"{tag}_hbp"].to_numpy(float)
        nums = {"kbb": kbb_num, "hr": raw[f"{tag}_hr"].to_numpy(float), "h": raw[f"{tag}_h"].to_numpy(float)}
        for stat in ["kbb", "hr", "h"]:
            denom_vec = den + pseudo
            vals = np.divide(
                nums[stat] + pseudo * priors[stat],
                denom_vec,
                out=priors[stat].copy(),
                where=denom_vec > 0,
            )
            out[f"sp_{stat}_{suffix}"] = vals
    return out


def best_starter(raw):
    variants = []
    long_grid = [0, 25, 50, 75, 100, 125, 150, 200]
    recent_grid = [0, 20, 40, 60, 80, 100, 125]
    for denom in ["pa_classified", "pa_all"]:
        for hbp in [False, True]:
            best = None
            for pl in long_grid:
                for pr in recent_grid:
                    feats = starter_features(raw, denom, hbp, pl, pr)
                    loss, detail = score(feats)
                    rec = {
                        "denominator": denom,
                        "subtract_hbp_in_kbb": hbp,
                        "pseudo_long": pl,
                        "pseudo_recent5": pr,
                        "loss": loss,
                        "detail": detail,
                    }
                    if best is None or loss < best["loss"]:
                        best = rec
            variants.append(best)
    return sorted(variants, key=lambda x: x["loss"])


def bp_current_table(pa):
    tg, _, bp = mu.game_tables(pa)
    bp = bp.copy()
    bp["game_ra9"] = np.where(bp["bp_outs"] > 0, bp["bp_runs"] * 27.0 / bp["bp_outs"], np.nan)
    return tg, bp


def precompute_bp_raw(tg, bp):
    games = target_games(tg)
    rows = []
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.game_date).normalize()
        ss = pd.Timestamp(year=d.year, month=1, day=1)
        bp_pre = bp[(bp["game_date"] < d) & (bp["game_date"] >= ss)]
        bp30 = bp_pre[bp_pre["game_date"] >= d - pd.Timedelta(days=30)]
        tg_pre = tg[(tg["game_date"] < d) & (tg["game_date"] >= ss)]
        tg30 = tg_pre[tg_pre["game_date"] >= d - pd.Timedelta(days=30)]
        rec = {"game_pk": r.game_pk}
        for tag, pool in [("long", bp_pre), ("30", bp30)]:
            team = pool[pool["pitching_team"].astype(str).eq(str(r.opponent))]
            rec[f"pool_runs_{tag}"] = float(pool["bp_runs"].sum())
            rec[f"pool_outs_{tag}"] = float(pool["bp_outs"].sum())
            rec[f"team_runs_{tag}"] = float(team["bp_runs"].sum())
            rec[f"team_outs_{tag}"] = float(team["bp_outs"].sum())
            vals_all = pool["game_ra9"].dropna()
            vals_team = team["game_ra9"].dropna()
            rec[f"game_lg_mean_{tag}"] = float(vals_all.mean()) if len(vals_all) else 4.5
            rec[f"game_team_sum_{tag}"] = float(vals_team.sum())
            rec[f"game_team_n_{tag}"] = float(len(vals_team))
        for tag, pool in [("long", tg_pre), ("30", tg30)]:
            lg = float(pool["runs_late"].mean()) if len(pool) else 0.0
            vals = pool[pool["opponent"].astype(str).eq(str(r.opponent))]["runs_late"]
            rec[f"late_lg_mean_{tag}"] = lg
            rec[f"late_team_sum_{tag}"] = float(vals.sum())
            rec[f"late_team_n_{tag}"] = float(len(vals))
        rows.append(rec)
    return pd.DataFrame(rows)


def bp_features(raw, mode, pseudo_long, pseudo_30):
    out = {}
    for tag, pseudo, feat in [("long", float(pseudo_long), "opp_bp_ra9_long_rel"), ("30", float(pseudo_30), "opp_bp_ra9_30d_rel")]:
        if mode == "reliever_pooled":
            pool_runs = raw[f"pool_runs_{tag}"].to_numpy(float)
            pool_outs = raw[f"pool_outs_{tag}"].to_numpy(float)
            lg = np.divide(pool_runs * 27.0, pool_outs, out=np.full(len(raw), 4.5), where=pool_outs > 0)
            team_runs = raw[f"team_runs_{tag}"].to_numpy(float)
            team_outs = raw[f"team_outs_{tag}"].to_numpy(float)
            den = team_outs + pseudo
            ra9 = np.divide(
                (team_runs + pseudo * lg / 27.0) * 27.0,
                den,
                out=lg.copy(),
                where=den > 0,
            )
        elif mode == "reliever_game_mean":
            lg = raw[f"game_lg_mean_{tag}"].to_numpy(float)
            n = raw[f"game_team_n_{tag}"].to_numpy(float)
            s = raw[f"game_team_sum_{tag}"].to_numpy(float)
            den = n + pseudo
            ra9 = np.divide(s + pseudo * lg, den, out=lg.copy(), where=den > 0)
        elif mode == "late_team_ra":
            lg = raw[f"late_lg_mean_{tag}"].to_numpy(float)
            n = raw[f"late_team_n_{tag}"].to_numpy(float)
            s = raw[f"late_team_sum_{tag}"].to_numpy(float)
            den = n + pseudo
            ra9 = np.divide(s + pseudo * lg, den, out=lg.copy(), where=den > 0)
        else:
            raise ValueError(mode)
        out[feat] = np.divide(ra9, lg, out=np.ones_like(ra9), where=np.abs(lg) > 1e-12)
    return out


def best_bp(raw):
    variants = []
    grids = {
        "reliever_pooled": ([0, 27, 54, 81, 108], [0, 18, 36, 54]),
        "reliever_game_mean": ([0, 1, 2, 3, 5, 7, 10], [0, 1, 2, 3, 5]),
        "late_team_ra": ([0, 2, 5, 10, 20], [0, 1, 2, 3, 5]),
    }
    for mode, (gl, g30) in grids.items():
        best = None
        for pl in gl:
            for pr in g30:
                feats = bp_features(raw, mode, pl, pr)
                loss, detail = score(feats)
                rec = {"mode": mode, "pseudo_long": pl, "pseudo_30d": pr, "loss": loss, "detail": detail}
                if best is None or loss < best["loss"]:
                    best = rec
        variants.append(best)
    return sorted(variants, key=lambda x: x["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", default="out_training")
    ap.add_argument("--out", default="state/mu_v1_2_sp_bp_definition_diagnostic.json")
    args = ap.parse_args()
    p = ROOT / args.training_dir / "kbo_pbp_2024_training.csv.gz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("frozen artifact SHA mismatch")
    df = pd.read_csv(p, low_memory=False)
    pa = prepare_pa_fixed(df)
    tg, bp = bp_current_table(pa)
    st_alt = starter_game_table(pa)
    starter_raw = precompute_starter_raw(tg, st_alt)
    bp_raw = precompute_bp_raw(tg, bp)
    sp = best_starter(starter_raw)
    bpres = best_bp(bp_raw)
    payload = {
        "version": "mu-v1.2-sp-bp-definition-diagnostic-v0.2-vectorized",
        "official_shadow_allowed": False,
        "training_rows": int(len(starter_raw)),
        "unique_games": int(len(starter_raw) // 2),
        "starter_variants_ranked": sp,
        "bullpen_variants_ranked": bpres,
        "note": "Only semantically plausible definitions were tested. Ranking is diagnostic, not permission to promote.",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

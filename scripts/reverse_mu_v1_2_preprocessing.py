from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "research" / "mu-v1.2" / "model_spec.json").read_text(encoding="utf-8"))
OUT = ROOT / "state" / "mu_v1_2_preprocessing_reverse.json"
TRAIN_START = pd.Timestamp("2024-05-01")

FROZEN = {}
for comp in SPEC["components"].values():
    for k, v in comp["scaler_mean"].items():
        FROZEN.setdefault(k, {})["mean"] = float(v)
    for k, v in comp["scaler_scale"].items():
        FROZEN.setdefault(k, {})["scale"] = float(v)


def moment_error(vals: np.ndarray, feat: str) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    m = float(vals.mean())
    s = float(vals.std(ddof=0))
    fm, fs = FROZEN[feat]["mean"], FROZEN[feat]["scale"]
    # Relative scale error matters more than raw magnitude because features have different units.
    em = abs(m - fm) / max(abs(fs), 1e-9)
    es = abs(s - fs) / max(abs(fs), 1e-9)
    return {"mean": m, "scale": s, "frozen_mean": fm, "frozen_scale": fs,
            "z_mean_abs_error": em, "relative_scale_error": es, "loss": em + es}


def shrink_mean(sum_x: float, n: float, prior_mean: float, pseudo_n: float) -> float:
    if n <= 0 and pseudo_n <= 0:
        return prior_mean
    return float((sum_x + pseudo_n * prior_mean) / (n + pseudo_n)) if n + pseudo_n > 0 else float(prior_mean)


def build_raw_rows(tg: pd.DataFrame, st: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    games = tg[(tg["game_date"] >= TRAIN_START) & (tg["game_date"].dt.year == 2024)].copy()
    games = games.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)
    st_map = st.set_index(["game_pk", "pitching_team"])["starter"].to_dict()
    rows = []
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.game_date).normalize()
        season_start = pd.Timestamp(year=d.year, month=1, day=1)
        tg_pre = tg[(tg["game_date"] < d) & (tg["game_date"] >= season_start)]
        tg30 = tg_pre[tg_pre["game_date"] >= d - pd.Timedelta(days=30)]
        if tg_pre.empty or tg30.empty:
            continue
        my = tg_pre[tg_pre["team"].astype(str).eq(str(r.team))]
        my30 = tg30[tg30["team"].astype(str).eq(str(r.team))]

        lg = {
            "f5_long": float(tg_pre["runs_f5"].mean()),
            "f5_30": float(tg30["runs_f5"].mean()),
            "late_long": float(tg_pre["runs_late"].mean()),
            "late_30": float(tg30["runs_late"].mean()),
            "full_long": float(tg_pre["runs_full"].mean()),
        }

        # Opposing starter identity from the current game; prior features only use games before d.
        opp_st = st_map.get((r.game_pk, r.opponent))
        s_pre_all = st[(st["game_date"] < d) & (st["game_date"] >= season_start)]
        s_player = s_pre_all[s_pre_all["starter"].astype(str).eq(str(opp_st))].sort_values("game_date") if opp_st is not None else s_pre_all.iloc[0:0]
        s5 = s_player.tail(5)

        def starter_suff(x: pd.DataFrame) -> dict:
            return {
                "pa": float(x["pa"].sum()),
                "kbb_num": float(x["k"].sum() - x["bb"].sum()),
                "hr": float(x["hr"].sum()),
                "h": float(x["h"].sum()),
                "starts": float(len(x)),
                "ip_sum": float(x["ip"].sum()),
            }

        def league_starter_prior() -> dict:
            pa_n = float(s_pre_all["pa"].sum())
            starts_n = float(len(s_pre_all))
            return {
                "kbb": float((s_pre_all["k"].sum() - s_pre_all["bb"].sum()) / pa_n) if pa_n else 0.0,
                "hr": float(s_pre_all["hr"].sum() / pa_n) if pa_n else 0.0,
                "h": float(s_pre_all["h"].sum() / pa_n) if pa_n else 0.0,
                "ip": float(s_pre_all["ip"].sum() / starts_n) if starts_n else 5.0,
            }

        bp_pre = bp[(bp["game_date"] < d) & (bp["game_date"] >= season_start)]
        bp30 = bp_pre[bp_pre["game_date"] >= d - pd.Timedelta(days=30)]
        obp = bp_pre[bp_pre["pitching_team"].astype(str).eq(str(r.opponent))]
        obp30 = bp30[bp30["pitching_team"].astype(str).eq(str(r.opponent))]
        lg_bp_ra9 = float(bp_pre["bp_runs"].sum() * 27.0 / bp_pre["bp_outs"].sum()) if bp_pre["bp_outs"].sum() else 4.5
        lg_bp30_ra9 = float(bp30["bp_runs"].sum() * 27.0 / bp30["bp_outs"].sum()) if bp30["bp_outs"].sum() else lg_bp_ra9

        home_team = r.team if int(r.is_home) else r.opponent
        park_games = tg_pre[(tg_pre["team"].astype(str).eq(str(home_team))) & (tg_pre["is_home"] == 1)]
        ids = park_games["game_pk"].unique()
        pg = tg_pre[tg_pre["game_pk"].isin(ids)]

        rec = {
            "game_pk": r.game_pk, "game_date": str(d.date()), "team": r.team,
            "opponent": r.opponent, "is_home": float(r.is_home), "opp_starter": opp_st,
            "team_n_long": float(len(my)), "team_n_30": float(len(my30)),
            "team_f5_sum_long": float(my["runs_f5"].sum()), "team_f5_sum_30": float(my30["runs_f5"].sum()),
            "team_late_sum_long": float(my["runs_late"].sum()), "team_late_sum_30": float(my30["runs_late"].sum()),
            **{f"lg_{k}": v for k, v in lg.items()},
            **{f"sp_long_{k}": v for k, v in starter_suff(s_player).items()},
            **{f"sp_recent_{k}": v for k, v in starter_suff(s5).items()},
            **{f"lg_sp_{k}": v for k, v in league_starter_prior().items()},
            "opp_bp_runs_long": float(obp["bp_runs"].sum()), "opp_bp_outs_long": float(obp["bp_outs"].sum()),
            "opp_bp_runs_30": float(obp30["bp_runs"].sum()), "opp_bp_outs_30": float(obp30["bp_outs"].sum()),
            "lg_bp_ra9_long": lg_bp_ra9, "lg_bp_ra9_30": lg_bp30_ra9,
            "park_total_runs": float(pg["runs_full"].sum()), "park_games": float(len(ids)),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def offense_features(raw: pd.DataFrame, pseudo_long: float, pseudo_30: float) -> dict[str, np.ndarray]:
    out = {}
    for phase, sum_col, lg_col in [
        ("f5", "team_f5_sum", "lg_f5"), ("late", "team_late_sum", "lg_late")
    ]:
        vals_long, vals_30 = [], []
        for r in raw.itertuples(index=False):
            lg_long = getattr(r, f"{lg_col}_long")
            lg30 = getattr(r, f"{lg_col}_30")
            mlong = shrink_mean(getattr(r, f"{sum_col}_long"), r.team_n_long, lg_long, pseudo_long)
            m30 = shrink_mean(getattr(r, f"{sum_col}_30"), r.team_n_30, lg30, pseudo_30)
            vals_long.append(mlong / lg_long if lg_long else 1.0)
            vals_30.append(m30 / lg30 if lg30 else 1.0)
        out[f"off_{phase}_long_ratio"] = np.asarray(vals_long)
        out[f"off_{phase}_30d_ratio"] = np.asarray(vals_30)
    return out


def sp_rate_features(raw: pd.DataFrame, pseudo_long_pa: float, pseudo_recent_pa: float) -> dict[str, np.ndarray]:
    out = {k: [] for k in ["sp_kbb_long","sp_kbb_recent5","sp_hr_long","sp_hr_recent5","sp_h_long","sp_h_recent5"]}
    for r in raw.itertuples(index=False):
        for tag, p in [("long", pseudo_long_pa), ("recent", pseudo_recent_pa)]:
            pa_n = getattr(r, f"sp_{tag}_pa")
            prior = {"kbb": r.lg_sp_kbb, "hr": r.lg_sp_hr, "h": r.lg_sp_h}
            nums = {"kbb": getattr(r, f"sp_{tag}_kbb_num"), "hr": getattr(r, f"sp_{tag}_hr"), "h": getattr(r, f"sp_{tag}_h")}
            suffix = "long" if tag == "long" else "recent5"
            for stat in ["kbb","hr","h"]:
                val = (nums[stat] + p * prior[stat]) / (pa_n + p) if pa_n + p > 0 else prior[stat]
                out[f"sp_{stat}_{suffix}"].append(float(val))
    return {k: np.asarray(v) for k, v in out.items()}


def sp_ip_features(raw: pd.DataFrame, pseudo_long_starts: float, pseudo_recent_starts: float) -> dict[str, np.ndarray]:
    out = {"sp_ip_long": [], "sp_ip_recent5": []}
    for r in raw.itertuples(index=False):
        for tag, p, feat in [("long",pseudo_long_starts,"sp_ip_long"),("recent",pseudo_recent_starts,"sp_ip_recent5")]:
            n = getattr(r, f"sp_{tag}_starts")
            s = getattr(r, f"sp_{tag}_ip_sum")
            prior = r.lg_sp_ip
            out[feat].append(shrink_mean(s, n, prior, p))
    return {k: np.asarray(v) for k, v in out.items()}


def bp_features(raw: pd.DataFrame, pseudo_long_outs: float, pseudo_30_outs: float) -> dict[str, np.ndarray]:
    a, b = [], []
    for r in raw.itertuples(index=False):
        def rel(tag: str, p: float, lg: float) -> float:
            runs = getattr(r, f"opp_bp_runs_{tag}")
            outs = getattr(r, f"opp_bp_outs_{tag}")
            prior_runs = p * lg / 27.0
            ra9 = (runs + prior_runs) * 27.0 / (outs + p) if outs + p > 0 else lg
            return ra9 / lg if lg else 1.0
        a.append(rel("long", pseudo_long_outs, r.lg_bp_ra9_long))
        b.append(rel("30", pseudo_30_outs, r.lg_bp_ra9_30))
    return {"opp_bp_ra9_long_rel": np.asarray(a), "opp_bp_ra9_30d_rel": np.asarray(b)}


def park_features(raw: pd.DataFrame, pseudo_games: float) -> np.ndarray:
    vals = []
    for r in raw.itertuples(index=False):
        lg = r.lg_full_long
        n = r.park_games
        # team_game has two rows per game, so total_runs is both teams' combined game runs.
        park_rate = (r.park_total_runs + 2.0 * pseudo_games * lg) / (2.0 * (n + pseudo_games)) if n + pseudo_games > 0 else lg
        vals.append(park_rate / lg if lg else 1.0)
    return np.asarray(vals)


def score_group(features: dict[str, np.ndarray]) -> tuple[float, dict]:
    detail = {k: moment_error(v, k) for k, v in features.items()}
    return float(sum(d["loss"] for d in detail.values())), detail


def grid1(values, fn):
    best = None
    all_best = []
    for x in values:
        loss, detail = score_group(fn(float(x)))
        rec = {"x": float(x), "loss": loss, "detail": detail}
        if best is None or loss < best["loss"]:
            best = rec
        all_best.append(rec)
    return best, sorted(all_best, key=lambda z: z["loss"])[:8]


def grid2(values_a, values_b, fn):
    best = None
    top = []
    for a in values_a:
        for b in values_b:
            loss, detail = score_group(fn(float(a), float(b)))
            rec = {"a": float(a), "b": float(b), "loss": loss, "detail": detail}
            if best is None or loss < best["loss"]:
                best = rec
            top.append(rec)
    return best, sorted(top, key=lambda z: z["loss"])[:10]


def main():
    df = mu.load_seasons(2024)
    pa = prepare_pa_fixed(df)
    tg, st, bp = mu.game_tables(pa)
    raw = build_raw_rows(tg, st, bp)

    # Broad, interpretable grids first. Follow-up can refine around the best candidates.
    off_vals_long = [0,1,2,3,4,5,7,10,12,15,20,25,30,40]
    off_vals_30 = [0,1,2,3,4,5,7,10,12,15,20]
    pa_vals_long = [0,10,20,30,40,50,75,100,125,150,200,250,300]
    pa_vals_recent = [0,10,20,30,40,50,60,75,100,125,150]
    start_vals = [0,0.5,1,1.5,2,3,4,5,7,10,15]
    bp_out_vals_long = [0,9,18,27,36,54,81,108,135,180]
    bp_out_vals_30 = [0,9,18,27,36,54,81,108]
    park_vals = [0,1,2,3,5,7,10,12,15,20,25,30]

    off_best, off_top = grid2(off_vals_long, off_vals_30, offense_features.__get__(None, object) if False else lambda a,b: offense_features(raw,a,b))
    spr_best, spr_top = grid2(pa_vals_long, pa_vals_recent, lambda a,b: sp_rate_features(raw,a,b))
    spi_best, spi_top = grid2(start_vals, start_vals, lambda a,b: sp_ip_features(raw,a,b))
    bp_best, bp_top = grid2(bp_out_vals_long, bp_out_vals_30, lambda a,b: bp_features(raw,a,b))
    park_best, park_top = grid1(park_vals, lambda x: {"park_factor": park_features(raw,x)})

    payload = {
        "version": "mu-v1.2-preprocessing-reverse-v0.1",
        "expected_train_rows": 1156,
        "raw_rows": int(len(raw)),
        "unique_games": int(raw["game_pk"].nunique()),
        "missing_current_starter_identity_rows": int(raw["opp_starter"].isna().sum()),
        "hypothesis": "league-prior pseudo-count shrinkage with no thin-history row drops",
        "best": {
            "offense_pseudo_games_long_30": off_best,
            "starter_rate_pseudo_pa_long_recent5": spr_best,
            "starter_ip_pseudo_starts_long_recent5": spi_best,
            "bullpen_pseudo_outs_long_30": bp_best,
            "park_pseudo_games": park_best,
        },
        "top_candidates": {
            "offense": off_top, "starter_rate": spr_top, "starter_ip": spi_top,
            "bullpen": bp_top, "park": park_top,
        },
        "warning": "Reverse-engineering diagnostic only. Do not activate official Shadow from these inferred priors."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

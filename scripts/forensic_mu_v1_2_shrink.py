from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
import run_mu_v1_2_pipeline as runner

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "research" / "mu-v1.2" / "model_spec.json").read_text(encoding="utf-8"))
OUT = ROOT / "state" / "mechanical_mu_shrink_forensics.json"
START = pd.Timestamp("2024-05-01")


def frozen(feat: str) -> tuple[float, float]:
    for comp in SPEC["components"].values():
        if feat in comp["features"]:
            return float(comp["scaler_mean"][feat]), float(comp["scaler_scale"][feat])
    raise KeyError(feat)


def objective(vals: np.ndarray, feat: str) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    m = float(vals.mean())
    s = float(vals.std(ddof=0))
    fm, fs = frozen(feat)
    # Dimensionless distance. Scale matching matters strongly for identifying shrink.
    obj = ((m - fm) / max(fs, 1e-9)) ** 2 + ((s - fs) / max(fs, 1e-9)) ** 2
    return obj, m, s


def best_grid(records: list[dict], feat: str, param: str, grid: list[float], fn) -> dict:
    best = None
    for p in grid:
        vals = np.asarray([fn(r, p) for r in records], dtype=float)
        obj, m, s = objective(vals, feat)
        row = {param: p, "objective": obj, "mean": m, "scale": s}
        if best is None or obj < best["objective"]:
            best = row
    fm, fs = frozen(feat)
    best["frozen_mean"] = fm
    best["frozen_scale"] = fs
    best["abs_mean_diff"] = abs(best["mean"] - fm)
    best["abs_scale_diff"] = abs(best["scale"] - fs)
    return best


def main() -> None:
    mu.prepare_pa = runner.prepare_pa_fixed
    df = mu.load_seasons(2025)
    pa = mu.prepare_pa(df)
    tg, st, bp = mu.game_tables(pa)
    st_map = st.set_index(["game_pk", "pitching_team"])["starter"].to_dict()

    recs: list[dict] = []
    rows = tg[tg["game_date"] >= START].sort_values(["game_date", "game_pk", "is_home"])
    for r in rows.itertuples(index=False):
        d = r.game_date
        season_start = pd.Timestamp(year=d.year, month=1, day=1)
        tg_pre = tg[(tg.game_date < d) & (tg.game_date >= season_start)]
        tg30 = tg_pre[tg_pre.game_date >= d - pd.Timedelta(days=30)]
        if tg_pre.empty:
            continue
        my = tg_pre[tg_pre.team.astype(str).eq(str(r.team))]
        my30 = tg30[tg30.team.astype(str).eq(str(r.team))]

        s_pre = st[(st.game_date < d) & (st.game_date >= season_start)]
        opp_starter = st_map.get((r.game_pk, r.opponent))
        mine_s = s_pre[s_pre.starter.astype(str).eq(str(opp_starter))].sort_values("game_date") if opp_starter is not None else s_pre.iloc[0:0]
        mine_s5 = mine_s.tail(5)

        bp_pre = bp[(bp.game_date < d) & (bp.game_date >= season_start)]
        bp30 = bp_pre[bp_pre.game_date >= d - pd.Timedelta(days=30)]
        mine_bp = bp_pre[bp_pre.pitching_team.astype(str).eq(str(r.opponent))]
        mine_bp30 = bp30[bp30.pitching_team.astype(str).eq(str(r.opponent))]

        def team_block(x: pd.DataFrame, metric: str) -> tuple[float, float]:
            return float(x[metric].sum()), float(len(x))

        def starter_block(x: pd.DataFrame) -> dict:
            return {
                "pa": float(x.pa.sum()),
                "kbb": float((x.k - x.bb).sum()),
                "h": float(x.h.sum()),
                "hr": float(x.hr.sum()),
                "ip_sum": float(x.ip.sum()),
                "starts": float(len(x)),
            }

        def league_starter_rate(metric: str) -> float:
            if s_pre.empty:
                return np.nan
            if metric == "ip":
                return float(s_pre.ip.mean())
            den = float(s_pre.pa.sum())
            if den <= 0:
                return np.nan
            if metric == "kbb":
                return float((s_pre.k.sum() - s_pre.bb.sum()) / den)
            return float(s_pre[metric].sum() / den)

        def bp_block(x: pd.DataFrame) -> tuple[float, float]:
            return float(x.bp_runs.sum()), float(x.bp_outs.sum())

        lf5 = float(tg_pre.runs_f5.mean())
        lf530 = float(tg30.runs_f5.mean()) if len(tg30) else lf5
        llate = float(tg_pre.runs_late.mean())
        llate30 = float(tg30.runs_late.mean()) if len(tg30) else llate
        league_bp_ra9 = float(bp_pre.bp_runs.sum() * 27 / bp_pre.bp_outs.sum()) if len(bp_pre) and bp_pre.bp_outs.sum() > 0 else np.nan
        league_bp30_ra9 = float(bp30.bp_runs.sum() * 27 / bp30.bp_outs.sum()) if len(bp30) and bp30.bp_outs.sum() > 0 else league_bp_ra9

        recs.append({
            "game_pk": r.game_pk,
            "date": str(d.date()),
            "is_home": int(r.is_home),
            "off_f5_long": team_block(my, "runs_f5"),
            "off_f5_30": team_block(my30, "runs_f5"),
            "off_late_long": team_block(my, "runs_late"),
            "off_late_30": team_block(my30, "runs_late"),
            "lf5": lf5, "lf530": lf530, "llate": llate, "llate30": llate30,
            "s_long": starter_block(mine_s), "s_recent": starter_block(mine_s5),
            "lkbb": league_starter_rate("kbb"), "lh": league_starter_rate("h"),
            "lhr": league_starter_rate("hr"), "lip": league_starter_rate("ip"),
            "bp_long": bp_block(mine_bp), "bp_30": bp_block(mine_bp30),
            "lbp": league_bp_ra9, "lbp30": league_bp30_ra9,
        })

    # Require a defined contemporaneous league prior; unlike the earlier audit,
    # player/team with zero personal history are retained and shrink fully to prior.
    recs = [r for r in recs if all(np.isfinite(r[k]) for k in ("lf5","lf530","llate","llate30","lkbb","lh","lhr","lip","lbp","lbp30"))]

    results = {}
    game_grid = [x / 2 for x in range(0, 81)]  # 0..40 by 0.5
    pa_grid = list(range(0, 501, 5))
    start_grid = [x / 2 for x in range(0, 41)]  # 0..20 by 0.5
    outs_grid = list(range(0, 1001, 10))

    def off_value(key: str, league_key: str):
        def _fn(r, g):
            runs, n = r[key]
            lg = r[league_key]
            return (runs + g * lg) / (n + g) / lg if (n + g) > 0 and lg > 0 else 1.0
        return _fn

    results["off_f5_long_ratio"] = best_grid(recs, "off_f5_long_ratio", "pseudo_games", game_grid, off_value("off_f5_long", "lf5"))
    results["off_f5_30d_ratio"] = best_grid(recs, "off_f5_30d_ratio", "pseudo_games", game_grid, off_value("off_f5_30", "lf530"))
    results["off_late_long_ratio"] = best_grid(recs, "off_late_long_ratio", "pseudo_games", game_grid, off_value("off_late_long", "llate"))
    results["off_late_30d_ratio"] = best_grid(recs, "off_late_30d_ratio", "pseudo_games", game_grid, off_value("off_late_30", "llate30"))

    for suffix, block in (("long", "s_long"), ("recent5", "s_recent")):
        for metric, league_key in (("kbb", "lkbb"), ("h", "lh"), ("hr", "lhr")):
            feat = f"sp_{metric}_{suffix}"
            def fn(r, p, block=block, metric=metric, league_key=league_key):
                b = r[block]
                return (b[metric] + p * r[league_key]) / (b["pa"] + p) if (b["pa"] + p) > 0 else r[league_key]
            results[feat] = best_grid(recs, feat, "pseudo_pa", pa_grid, fn)

        feat = f"sp_ip_{suffix}"
        def ip_fn(r, p, block=block):
            b = r[block]
            return (b["ip_sum"] + p * r["lip"]) / (b["starts"] + p) if (b["starts"] + p) > 0 else r["lip"]
        results[feat] = best_grid(recs, feat, "pseudo_starts", start_grid, ip_fn)

    for suffix, block, league_key in (("long_rel", "bp_long", "lbp"), ("30d_rel", "bp_30", "lbp30")):
        feat = f"opp_bp_ra9_{suffix}"
        def bp_fn(r, p, block=block, league_key=league_key):
            runs, outs = r[block]
            lg = r[league_key]
            team_ra9 = (runs * 27 + p * lg) / (outs + p) if (outs + p) > 0 else lg
            return team_ra9 / lg if lg > 0 else 1.0
        results[feat] = best_grid(recs, feat, "pseudo_outs", outs_grid, bp_fn)

    payload = {
        "version": "mu-v1.2-shrink-forensics-v0.1",
        "rows": len(recs),
        "purpose": "Infer simple integer/half-integer shrink hypotheses from frozen scaler moments. This is not proof of the original preprocessing and cannot authorize exact v1.2 Shadow by itself.",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

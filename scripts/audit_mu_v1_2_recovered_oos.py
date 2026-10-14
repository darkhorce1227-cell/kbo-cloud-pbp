from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as base
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as engine
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
PUBLISHED_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_recovered_challenger_oos.json"

HISTORY_START = pd.Timestamp("2024-01-01")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-10-01")

# Recovered/near-recovered constants from 2024-2025 frozen-moment forensics only.
OFF_F5_LONG_P = 5.2
OFF_F5_30D_P = 6.1
OFF_LATE_LONG_P = 13.75
OFF_LATE_30D_P = 6.0
SP_RATE_LONG_P = 150.0
SP_RATE_RECENT_P = 150.0
SP_IP_LONG_P = 5.0
SP_IP_RECENT_P = 5.0
BP_LONG_OUTS_P = 360.0
BP_RECENT_OUTS_P = 54.0
PARK_GAMES_P = 5.5


def safe_ratio(num: float, den: float, fallback: float = 1.0) -> float:
    return float(num / den) if den and np.isfinite(den) and abs(den) > 1e-12 else float(fallback)


def load_tables():
    base.prepare_pa = runner.prepare_pa_fixed
    raw = base.load_seasons(2026)
    pa = runner.prepare_pa_fixed(raw)
    tg, st_base, _ = make_team_game(pa)

    # Exact all-BF starter event counts; merge IP from the base starter game table.
    st = starter_game_table(pa).copy()
    ip_map = {
        (str(r.game_pk), str(r.pitching_team)): float(r.ip)
        for r in st_base.itertuples(index=False)
    }
    st["ip"] = [ip_map.get((str(r.game_pk), str(r.pitching_team)), np.nan) for r in st.itertuples(index=False)]
    if st["ip"].isna().any():
        raise RuntimeError(f"starter IP merge missing rows={int(st['ip'].isna().sum())}")

    # Semantic diagnostic winner: all pitching from inning 6 onward.
    bp6 = bp_game_tables(pa, tg)["inning6plus"].copy()
    return tg, st, bp6


def shrink_mean(total: float, n: float, prior: float, pseudo: float) -> float:
    den = float(n) + float(pseudo)
    return float((float(total) + float(pseudo) * float(prior)) / den) if den > 0 else float(prior)


def build_features(tg: pd.DataFrame, st: pd.DataFrame, bp6: pd.DataFrame, *, row) -> dict[str, float]:
    d = pd.Timestamp(row.game_date).normalize()
    team = str(row.team)
    opp = str(row.opponent)

    # Continuous team/offense history across seasons.
    pre = tg[(tg.game_date < d) & (tg.game_date >= HISTORY_START)]
    if pre.empty:
        raise RuntimeError(f"no team history before {d.date()}")
    pre30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
    my = pre[pre.team.astype(str).eq(team)]
    my30 = pre30[pre30.team.astype(str).eq(team)]
    if my30.empty:
        my30 = my.tail(min(30, len(my)))

    lf5 = float(pre.runs_f5.mean())
    llate = float(pre.runs_late.mean())
    lf5_30 = float(pre30.runs_f5.mean()) if len(pre30) else lf5
    llate_30 = float(pre30.runs_late.mean()) if len(pre30) else llate

    out = {
        "off_f5_long_ratio": safe_ratio(shrink_mean(my.runs_f5.sum(), len(my), lf5, OFF_F5_LONG_P), lf5),
        "off_f5_30d_ratio": safe_ratio(shrink_mean(my30.runs_f5.sum(), len(my30), lf5_30, OFF_F5_30D_P), lf5_30),
        "off_late_long_ratio": safe_ratio(shrink_mean(my.runs_late.sum(), len(my), llate, OFF_LATE_LONG_P), llate),
        "off_late_30d_ratio": safe_ratio(shrink_mean(my30.runs_late.sum(), len(my30), llate_30, OFF_LATE_30D_P), llate_30),
        "league_f5_30d": lf5_30,
        "league_late_30d": llate_30,
        "is_home": float(row.is_home),
    }

    # Current opposing starter identity, then continuous pregame starter history.
    current = st[(st.game_pk.astype(str) == str(row.game_pk)) & (st.pitching_team.astype(str) == opp)]
    if len(current) != 1:
        raise RuntimeError(f"starter identity rows={len(current)} game={row.game_pk} opp={opp}")
    sid = str(current.iloc[0].starter)
    sp_pre = st[(st.game_date < d) & (st.game_date >= HISTORY_START)]
    mine = sp_pre[sp_pre.starter.astype(str).eq(sid)].sort_values(["game_date", "game_pk"])
    recent = mine.tail(5)

    league_pa = float(sp_pre.pa_all.sum())
    league_kbb = safe_ratio(float(sp_pre.k.sum() - sp_pre.bb.sum() - sp_pre.hbp.sum()), league_pa, 0.0)
    league_hr = safe_ratio(float(sp_pre.hr.sum()), league_pa, 0.023)
    league_h = safe_ratio(float(sp_pre.h.sum()), league_pa, 0.235)
    league_ip = float(sp_pre.ip.mean()) if len(sp_pre) else 5.0

    def rate(x: pd.DataFrame, stat: str, prior: float, pseudo: float) -> float:
        den = float(x.pa_all.sum())
        if stat == "kbb":
            num = float(x.k.sum() - x.bb.sum() - x.hbp.sum())
        else:
            num = float(x[stat].sum())
        return shrink_mean(num, den, prior, pseudo)

    kbb_long = rate(mine, "kbb", league_kbb, SP_RATE_LONG_P)
    hr_long = rate(mine, "hr", league_hr, SP_RATE_LONG_P)
    h_long = rate(mine, "h", league_h, SP_RATE_LONG_P)
    ip_long = shrink_mean(float(mine.ip.sum()), len(mine), league_ip, SP_IP_LONG_P)

    out.update({
        "sp_kbb_long": kbb_long,
        "sp_kbb_recent5": rate(recent, "kbb", kbb_long, SP_RATE_RECENT_P),
        "sp_hr_long": hr_long,
        "sp_hr_recent5": rate(recent, "hr", hr_long, SP_RATE_RECENT_P),
        "sp_h_long": h_long,
        "sp_h_recent5": rate(recent, "h", h_long, SP_RATE_RECENT_P),
        "sp_ip_long": ip_long,
        "sp_ip_recent5": shrink_mean(float(recent.ip.sum()), len(recent), ip_long, SP_IP_RECENT_P),
    })

    # Bullpen winner from semantic audit: inning6+, season-reset long history,
    # calendar-30 recent; recent prior is team's shrunk long RA9.
    bp_all_pre = bp6[bp6.game_date < d]
    bp_season = bp_all_pre[bp_all_pre.game_date.dt.year == d.year]
    bp_pre = bp_season if len(bp_season) else bp_all_pre
    bpo = bp_pre[bp_pre.pitching_team.astype(str).eq(opp)]
    lg_outs = float(bp_pre.outs.sum())
    lg_ra9 = safe_ratio(float(bp_pre.runs.sum()) * 27.0, lg_outs, 4.5)
    team_long = safe_ratio(
        (float(bpo.runs.sum()) + BP_LONG_OUTS_P * lg_ra9 / 27.0) * 27.0,
        float(bpo.outs.sum()) + BP_LONG_OUTS_P,
        lg_ra9,
    )

    bp30 = bp6[(bp6.game_date < d) & (bp6.game_date >= d - pd.Timedelta(days=30))]
    bpo30 = bp30[bp30.pitching_team.astype(str).eq(opp)]
    lg30_outs = float(bp30.outs.sum())
    lg30 = safe_ratio(float(bp30.runs.sum()) * 27.0, lg30_outs, lg_ra9)
    team30 = safe_ratio(
        (float(bpo30.runs.sum()) + BP_RECENT_OUTS_P * team_long / 27.0) * 27.0,
        float(bpo30.outs.sum()) + BP_RECENT_OUTS_P,
        team_long,
    )
    out["opp_bp_ra9_long_rel"] = safe_ratio(team_long, lg_ra9)
    out["opp_bp_ra9_30d_rel"] = safe_ratio(team30, lg30)

    # Continuous-history home-park factor.
    home_team = team if int(row.is_home) else opp
    park_home_rows = pre[(pre.team.astype(str).eq(home_team)) & (pre.is_home == 1)]
    ids = park_home_rows.game_pk.unique()
    league_full = float(pre.runs_full.mean())
    pg = pre[pre.game_pk.isin(ids)]
    if len(ids) and league_full > 0:
        park_rate = (float(pg.runs_full.sum()) + 2.0 * PARK_GAMES_P * league_full) / (2.0 * (float(len(ids)) + PARK_GAMES_P))
        out["park_factor"] = park_rate / league_full
    else:
        out["park_factor"] = 1.0
    return out


def make_oos_rows(tg, st, bp6):
    target = tg[(tg.game_date >= OOS_START) & (tg.game_date <= OOS_END)].sort_values(["game_date", "game_pk", "is_home"])
    rows = []
    for r in target.itertuples(index=False):
        f = build_features(tg, st, bp6, row=r)
        f.update({
            "game_pk": str(r.game_pk),
            "game_date": str(pd.Timestamp(r.game_date).date()),
            "team": str(r.team),
            "opponent": str(r.opponent),
            "is_home": float(r.is_home),
            "runs_f5": float(r.runs_f5),
            "runs_late": float(r.runs_late),
            "runs_full": float(r.runs_full),
        })
        rows.append(f)
    return pd.DataFrame(rows)


def mean_mu(rows: pd.DataFrame, model: dict) -> dict[str, float]:
    f5, full = [], []
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away = x[x.is_home == 0].iloc[0].to_dict()
        home = x[x.is_home == 1].iloc[0].to_dict()
        pred = engine.score(model, away, home)
        f5.extend([float(pred["f5_away"]), float(pred["f5_home"])])
        full.extend([float(pred["full_away"]), float(pred["full_home"])])
    return {"f5": float(np.mean(f5)), "full": float(np.mean(full))}


def main():
    tg, st, bp6 = load_tables()
    rows = make_oos_rows(tg, st, bp6)
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    got = engine.evaluate(rows, model)
    mm = mean_mu(rows, model)

    pub = published["oos_2026"]
    mapped = {
        "f5": {
            "mae": got["f5"]["team_score_mae"],
            "nb_nll": got["f5"]["nb_nll"],
            "wdl_brier": got["f5"]["wdl_brier_mean3"],
            "wdl_logloss": got["f5"]["wdl_logloss"],
            "decided_side": got["f5"]["decided_side_hit"],
            "mean_mu": mm["f5"],
        },
        "full": {
            "mae": got["full"]["team_score_mae"],
            "nb_nll": got["full"]["nb_nll"],
            "wdl_brier": got["full"]["wdl_brier_mean3"],
            "wdl_logloss": got["full"]["wdl_logloss"],
            "decided_side": got["full"]["decided_side_hit"],
            "mean_mu": mm["full"],
        },
    }
    pub_side = published["decided_side_accuracy_2026"]
    deltas = {}
    for kind in ["f5", "full"]:
        p = pub[kind]["hybrid"]
        deltas[kind] = {
            "mae": mapped[kind]["mae"] - float(p["mae"]),
            "nb_nll": mapped[kind]["nb_nll"] - float(p["nb_nll"]),
            "wdl_brier": mapped[kind]["wdl_brier"] - float(p["wdl_brier"]),
            "wdl_logloss": mapped[kind]["wdl_logloss"] - float(p["wdl_logloss"]),
            "mean_mu": mapped[kind]["mean_mu"] - float(p["mean_mu"]),
            "decided_side": mapped[kind]["decided_side"] - float(pub_side[kind]),
        }

    payload = {
        "version": "mu-v1.2-recovered-challenger-oos-v0.1",
        "status": "CHALLENGER_DIAGNOSTIC",
        "official_shadow_allowed": False,
        "oos_team_games": int(len(rows)),
        "oos_games": int(rows.game_pk.nunique()),
        "expected_team_games": 1368,
        "expected_games": 684,
        "constants": {
            "off_f5_long_p": OFF_F5_LONG_P, "off_f5_30d_p": OFF_F5_30D_P,
            "off_late_long_p": OFF_LATE_LONG_P, "off_late_30d_p": OFF_LATE_30D_P,
            "sp_rate_long_p": SP_RATE_LONG_P, "sp_rate_recent_p": SP_RATE_RECENT_P,
            "sp_ip_long_p": SP_IP_LONG_P, "sp_ip_recent_p": SP_IP_RECENT_P,
            "bp_definition": "inning6plus", "bp_scope": "season_reset",
            "bp_long_outs_p": BP_LONG_OUTS_P, "bp_recent_outs_p": BP_RECENT_OUTS_P,
            "park_games_p": PARK_GAMES_P,
        },
        "recovered_2026_oos": mapped,
        "published_v1_2_hybrid_2026_oos": {
            "f5": {**pub["f5"]["hybrid"], "decided_side": pub_side["f5"]},
            "full": {**pub["full"]["hybrid"], "decided_side": pub_side["full"]},
        },
        "delta_recovered_minus_published": deltas,
        "note": "Final v1.2 model coefficients/scalers/gamma/NB alpha are unchanged. This audit changes preprocessing only and cannot activate Official Shadow.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

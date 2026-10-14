from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import kbo_pbp
from kbo_pbp import events as eventlib

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
START_SEASON = 2024
TRAIN_START = pd.Timestamp("2024-05-01")
HITS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {"strikeout", "field_out", "sac_bunt", "sac_fly"}


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def load_seasons(last_year: int) -> pd.DataFrame:
    frames = []
    for year in range(START_SEASON, last_year + 1):
        errors = kbo_pbp.download(year)
        if errors:
            raise RuntimeError(f"{year}: kbo_pbp.download failures={len(errors)}")
        df = kbo_pbp.season(year)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"{year}: empty/non-dataframe season")
        frames.append(df.copy())
    out = pd.concat(frames, ignore_index=True)
    need = {
        "game_pk","game_date","home_team","away_team","inning","inning_topbot",
        "at_bat_number","pitch_number","pitcher","events","outs_when_up",
        "home_score","away_score","post_home_score","post_away_score"
    }
    missing = sorted(need - set(out.columns))
    if missing:
        raise RuntimeError(f"missing required PBP columns: {missing}")
    dup = int(out.duplicated(["game_pk","at_bat_number","pitch_number"]).sum())
    if dup:
        raise RuntimeError(f"duplicate stable pitch keys={dup}")
    return out


def prepare_pa(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["game_date"] = _to_date(x["game_date"])
    pa = eventlib.plate_appearances(x)
    pa["game_date"] = _to_date(pa["game_date"])
    top = pa["inning_topbot"].astype(str).str.lower().str.startswith("top")
    pa["batting_team"] = np.where(top, pa["away_team"], pa["home_team"])
    pa["pitching_team"] = np.where(top, pa["home_team"], pa["away_team"])
    for c in ("home_score","away_score","post_home_score","post_away_score","outs_when_up","inning"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")
    pa["runs_scored"] = (
        pa["post_home_score"].fillna(pa["home_score"]) +
        pa["post_away_score"].fillna(pa["away_score"]) -
        pa["home_score"].fillna(0) - pa["away_score"].fillna(0)
    ).clip(lower=0)

    pa["_half"] = (
        pa["game_pk"].astype(str) + "|" +
        pa["inning"].astype("Int64").astype(str) + "|" +
        pa["inning_topbot"].astype(str)
    )
    pa = pa.sort_values(["game_date","game_pk","inning","inning_topbot","at_bat_number"]).reset_index(drop=True)
    nxt_half = pa["_half"].shift(-1)
    nxt_outs = pa["outs_when_up"].shift(-1)
    same_half = pa["_half"].eq(nxt_half)
    out_delta = np.where(
        same_half,
        (nxt_outs - pa["outs_when_up"]).clip(lower=0),
        (3 - pa["outs_when_up"]).clip(lower=0),
    )
    walkoff_like = (~same_half) & (pa["inning"] >= 9) & (pa["inning_topbot"].astype(str).str.lower().str.startswith("bot"))
    event_outs = pa["event"].isin(OUT_EVENTS).astype(int)
    event_outs += pa["event"].eq("double_play").astype(int) * 2
    event_outs += pa["event"].eq("triple_play").astype(int) * 3
    out_delta = np.where(walkoff_like, np.minimum(out_delta, event_outs), out_delta)
    pa["outs_recorded"] = pd.to_numeric(out_delta, errors="coerce").fillna(0).clip(lower=0, upper=3)
    return pa


def game_tables(pa: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    g = pa.groupby(["game_pk","game_date","home_team","away_team"], as_index=False).agg(
        home_final=("post_home_score","max"),
        away_final=("post_away_score","max"),
    )
    f5 = pa[pa["inning"] <= 5].groupby("game_pk", as_index=False).agg(
        home_f5=("post_home_score","max"),
        away_f5=("post_away_score","max"),
    )
    g = g.merge(f5, on="game_pk", how="left").fillna({"home_f5":0,"away_f5":0})
    home = pd.DataFrame({
        "game_pk":g.game_pk,"game_date":g.game_date,"team":g.home_team,"opponent":g.away_team,
        "is_home":1,"runs_f5":g.home_f5,"runs_full":g.home_final,
    })
    away = pd.DataFrame({
        "game_pk":g.game_pk,"game_date":g.game_date,"team":g.away_team,"opponent":g.home_team,
        "is_home":0,"runs_f5":g.away_f5,"runs_full":g.away_final,
    })
    team_game = pd.concat([home, away], ignore_index=True)
    team_game["runs_late"] = (team_game["runs_full"] - team_game["runs_f5"]).clip(lower=0)

    first = (
        pa.sort_values(["game_date","game_pk","inning","at_bat_number"])
        .groupby(["game_pk","pitching_team"], as_index=False)
        .first()[["game_pk","pitching_team","pitcher","game_date"]]
        .rename(columns={"pitcher":"starter"})
    )
    p = pa.merge(first[["game_pk","pitching_team","starter"]], on=["game_pk","pitching_team"], how="left")
    p["is_starter_pitch"] = p["pitcher"].astype(str).eq(p["starter"].astype(str))
    p["is_k"] = p["event"].eq("strikeout").astype(int)
    p["is_bb"] = p["event"].eq("walk").astype(int)
    p["is_hr"] = p["event"].eq("home_run").astype(int)
    p["is_h"] = p["event"].isin(HITS).astype(int)
    p["pa1"] = p["event"].notna().astype(int)
    st = (
        p[p["is_starter_pitch"]]
        .groupby(["game_pk","game_date","pitching_team","starter"], as_index=False)
        .agg(pa=("pa1","sum"),k=("is_k","sum"),bb=("is_bb","sum"),hr=("is_hr","sum"),
             h=("is_h","sum"),outs=("outs_recorded","sum"))
    )
    st["ip"] = st["outs"] / 3.0
    bp = (
        p[~p["is_starter_pitch"]]
        .groupby(["game_pk","game_date","pitching_team"], as_index=False)
        .agg(bp_runs=("runs_scored","sum"),bp_outs=("outs_recorded","sum"))
    )
    return team_game, st, bp


def _rate(num: float, den: float, fallback: float = float("nan")) -> float:
    return float(num / den) if den and den > 0 else fallback


def build_feature_row(
    team_game: pd.DataFrame, starters: pd.DataFrame, bullpens: pd.DataFrame,
    *, d: pd.Timestamp, team: str, opponent: str, opposing_starter: Any,
    is_home: int, park_factor_override: float | None = None,
) -> dict[str,float]:
    season_start = pd.Timestamp(year=d.year, month=1, day=1)
    tg_pre = team_game[(team_game["game_date"] < d) & (team_game["game_date"] >= season_start)]
    tg30 = tg_pre[tg_pre["game_date"] >= d - pd.Timedelta(days=30)]
    if tg_pre.empty or tg30.empty:
        raise RuntimeError(f"insufficient team-game history before {d.date()}")
    league_f5_long = tg_pre["runs_f5"].mean()
    league_f5_30 = tg30["runs_f5"].mean()
    league_late_long = tg_pre["runs_late"].mean()
    league_late_30 = tg30["runs_late"].mean()
    my = tg_pre[tg_pre["team"].astype(str).eq(str(team))]
    my30 = tg30[tg30["team"].astype(str).eq(str(team))]
    if len(my) < 10 or len(my30) < 3:
        raise RuntimeError(f"insufficient offense history for {team}")
    feats: dict[str,float] = {
        "off_f5_long_ratio": _rate(my["runs_f5"].mean(), league_f5_long),
        "off_f5_30d_ratio": _rate(my30["runs_f5"].mean(), league_f5_30),
        "off_late_long_ratio": _rate(my["runs_late"].mean(), league_late_long),
        "off_late_30d_ratio": _rate(my30["runs_late"].mean(), league_late_30),
        "league_f5_30d": float(league_f5_30),
        "league_late_30d": float(league_late_30),
        "is_home": float(is_home),
    }
    s_all = starters[
        (starters["game_date"] < d) &
        (starters["game_date"] >= season_start) &
        starters["starter"].astype(str).eq(str(opposing_starter))
    ].sort_values("game_date")
    if s_all.empty:
        raise RuntimeError(f"no prior starts for starter={opposing_starter}")
    s5 = s_all.tail(5)
    def svals(x: pd.DataFrame, suffix: str) -> None:
        pa_n = float(x["pa"].sum())
        feats[f"sp_kbb_{suffix}"] = _rate(float(x["k"].sum()-x["bb"].sum()), pa_n, 0.0)
        feats[f"sp_hr_{suffix}"] = _rate(float(x["hr"].sum()), pa_n, 0.0)
        feats[f"sp_h_{suffix}"] = _rate(float(x["h"].sum()), pa_n, 0.0)
        feats[f"sp_ip_{suffix}"] = float(x["ip"].mean())
    svals(s_all, "long")
    svals(s5, "recent5")

    bp_pre = bullpens[(bullpens["game_date"] < d) & (bullpens["game_date"] >= season_start)]
    bp30 = bp_pre[bp_pre["game_date"] >= d - pd.Timedelta(days=30)]
    opp_bp = bp_pre[bp_pre["pitching_team"].astype(str).eq(str(opponent))]
    opp_bp30 = bp30[bp30["pitching_team"].astype(str).eq(str(opponent))]
    def bp_ra9(x: pd.DataFrame) -> float:
        return _rate(float(x["bp_runs"].sum())*27.0, float(x["bp_outs"].sum()))
    feats["opp_bp_ra9_long_rel"] = _rate(bp_ra9(opp_bp), bp_ra9(bp_pre))
    feats["opp_bp_ra9_30d_rel"] = _rate(bp_ra9(opp_bp30), bp_ra9(bp30))

    if park_factor_override is not None:
        feats["park_factor"] = float(park_factor_override)
    else:
        home_team = team if is_home else opponent
        park_games = team_game[
            (team_game["game_date"] < d) &
            (team_game["game_date"] >= season_start) &
            (team_game["team"].astype(str).eq(str(home_team))) &
            (team_game["is_home"] == 1)
        ]
        if park_games.empty:
            feats["park_factor"] = 1.0
        else:
            ids = park_games["game_pk"].unique()
            pg = team_game[team_game["game_pk"].isin(ids)]
            total_runs = float(pg["runs_full"].sum())
            n_games = float(len(ids))
            league_rate = float(tg_pre["runs_full"].mean())
            pseudo_games = 20.0
            park_rate = (total_runs + 2*pseudo_games*league_rate) / (2*(n_games+pseudo_games))
            feats["park_factor"] = _rate(park_rate, league_rate, 1.0)
    return feats


def score_mu(feats_away: dict[str,float], feats_home: dict[str,float], spec: dict) -> dict[str,float]:
    def comp(name: str, f: dict[str,float]) -> float:
        s = spec["components"][name]
        zsum = float(s["params"]["const"])
        for k in s["features"]:
            z = (float(f[k]) - float(s["scaler_mean"][k])) / float(s["scaler_scale"][k])
            zsum += float(s["params"][k]) * z
        return float(f[s["offset"]]) * math.exp(zsum)
    sf_a, sf_h = comp("simple_f5", feats_away), comp("simple_f5", feats_home)
    cf_a, cf_h = comp("core_f5", feats_away), comp("core_f5", feats_home)
    f5_total = sf_a + sf_h
    g5 = float(spec["gamma"]["f5"])
    share_a = cf_a**g5 / (cf_a**g5 + cf_h**g5)
    f5_a, f5_h = f5_total*share_a, f5_total*(1-share_a)
    sl_a, sl_h = comp("simple_late", feats_away), comp("simple_late", feats_home)
    cl_a, cl_h = comp("core_late", feats_away), comp("core_late", feats_home)
    full_total = sf_a + sf_h + sl_a + sl_h
    cfa, cfh = cf_a + cl_a, cf_h + cl_h
    gf = float(spec["gamma"]["full"])
    share_full_a = cfa**gf / (cfa**gf + cfh**gf)
    return {
        "f5_away":f5_a,"f5_home":f5_h,"f5_total":f5_total,
        "full_away":full_total*share_full_a,
        "full_home":full_total*(1-share_full_a),
        "full_total":full_total,
    }


def training_rows(team_game: pd.DataFrame, starters: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    games = team_game[(team_game["game_date"] >= TRAIN_START) & (team_game["game_date"].dt.year == 2024)]
    st_map = starters.set_index(["game_pk","pitching_team"])["starter"].to_dict()
    for r in games.itertuples(index=False):
        opp_st = st_map.get((r.game_pk, r.opponent))
        if opp_st is None:
            continue
        try:
            f = build_feature_row(
                team_game, starters, bp, d=r.game_date, team=r.team, opponent=r.opponent,
                opposing_starter=opp_st, is_home=int(r.is_home)
            )
        except Exception:
            continue
        f.update({"game_pk":r.game_pk,"game_date":str(r.game_date.date()),"team":r.team})
        rows.append(f)
    return pd.DataFrame(rows)


def audit(out_path: Path) -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    df = load_seasons(2024)
    pa = prepare_pa(df)
    tg, st, bp = game_tables(pa)
    tr = training_rows(tg, st, bp)
    feature_names = sorted({k for c in spec["components"].values() for k in c["features"]})
    checks = {}
    max_mean = max_scale = 0.0
    for feat in feature_names:
        if feat not in tr.columns:
            checks[feat] = {"status":"MISSING"}
            continue
        vals = pd.to_numeric(tr[feat], errors="coerce").dropna().to_numpy(float)
        cand_mean = float(vals.mean()) if len(vals) else float("nan")
        cand_scale = float(vals.std(ddof=0)) if len(vals) else float("nan")
        refs = [(float(c["scaler_mean"][feat]), float(c["scaler_scale"][feat]))
                for c in spec["components"].values() if feat in c["features"]]
        ref_mean, ref_scale = refs[0]
        dm, ds = abs(cand_mean-ref_mean), abs(cand_scale-ref_scale)
        max_mean, max_scale = max(max_mean, dm), max(max_scale, ds)
        checks[feat] = {
            "n":len(vals),"candidate_mean":cand_mean,"frozen_mean":ref_mean,"abs_mean_diff":dm,
            "candidate_scale":cand_scale,"frozen_scale":ref_scale,"abs_scale_diff":ds,
        }
    exact = bool(len(tr) > 0 and max_mean <= 1e-6 and max_scale <= 1e-6)
    payload = {
        "version":"mu-v1.2-feature-rebuild-v0.1",
        "status":"EXACT_PARITY_PASS" if exact else "UNVERIFIED_PREPROCESSING",
        "training_rows":int(len(tr)),
        "max_abs_mean_diff":max_mean,
        "max_abs_scale_diff":max_scale,
        "checks":checks,
        "note":"Official Mechanical Shadow is permitted only when status=EXACT_PARITY_PASS.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return payload


def shadow(input_path: Path, out_path: Path, audit_path: Path) -> dict:
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "EXACT_PARITY_PASS":
        payload = {
            "status":"UNAVAILABLE",
            "reason":"mu-v1.2 preprocessing parity gate has not passed",
            "audit_status":audit_payload.get("status"),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return payload
    inp = json.loads(input_path.read_text(encoding="utf-8"))
    target = pd.Timestamp(inp["target_date"]).normalize()
    df = load_seasons(target.year)
    df["_d"] = _to_date(df["game_date"])
    pa = prepare_pa(df[df["_d"] < target].drop(columns=["_d"]))
    tg, st, bp = game_tables(pa)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    games_out = []
    for g in inp.get("games", []):
        af = build_feature_row(
            tg,st,bp,d=target,team=g["away_team"],opponent=g["home_team"],
            opposing_starter=g["home_starter"],is_home=0,park_factor_override=g.get("park_factor"),
        )
        hf = build_feature_row(
            tg,st,bp,d=target,team=g["home_team"],opponent=g["away_team"],
            opposing_starter=g["away_starter"],is_home=1,park_factor_override=g.get("park_factor"),
        )
        games_out.append({
            "game_pk":g.get("game_pk"),"away_team":g["away_team"],"home_team":g["home_team"],
            "away_features":af,"home_features":hf,"mechanical_mu":score_mu(af,hf,spec)
        })
    payload = {"status":"ACTIVE","model":"mu-v1.2-research","target_date":str(target.date()),"games":games_out}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--out", default="state/mechanical_mu_pipeline_audit.json")
    s = sub.add_parser("shadow")
    s.add_argument("--input", default="state/mechanical_mu_input.json")
    s.add_argument("--audit", default="state/mechanical_mu_pipeline_audit.json")
    s.add_argument("--out", default="state/mechanical_mu_shadow.json")
    args = ap.parse_args()
    payload = audit(ROOT / args.out) if args.cmd == "audit" else shadow(ROOT / args.input, ROOT / args.out, ROOT / args.audit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

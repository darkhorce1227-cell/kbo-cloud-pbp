from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import build_mu_v1_2_features as base
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
CHALLENGER_PATH = ROOT / "research" / "mu-v1.2" / "recovered_challenger.json"
BP_LONG_OUTS_P = 360.0
BP_RECENT_WINDOW_DAYS = 36
BP_RECENT_OUTS_P = 25.5
LEAGUE_ENV_30D_LONG_PRIOR_P = 17.0


def _load_history(target: pd.Timestamp):
    base.prepare_pa = runner.prepare_pa_fixed
    raw = base.load_seasons(int(target.year))
    raw["_d"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.normalize()
    raw = raw[raw["_d"] < target].drop(columns=["_d"])
    pa = runner.prepare_pa_fixed(raw)
    tg, st_base, _ = make_team_game(pa)
    st = starter_game_table(pa).copy()
    ip_map = {(str(r.game_pk), str(r.pitching_team)): float(r.ip) for r in st_base.itertuples(index=False)}
    st["ip"] = [ip_map.get((str(r.game_pk), str(r.pitching_team)), np.nan) for r in st.itertuples(index=False)]
    if st["ip"].isna().any():
        raise RuntimeError(f"starter IP merge missing rows={int(st['ip'].isna().sum())}")
    bps = bp_game_tables(pa, tg)
    bp6 = bps["inning6plus"].copy()
    bprel = bps["reliever_stint"].copy()
    return tg, st, bp6, bprel


def _inject_current_starter(st, *, game_pk, d, pitching_team, starter):
    row = {c: 0 for c in st.columns}
    row.update({"game_pk": str(game_pk), "game_date": d, "pitching_team": str(pitching_team), "starter": str(starter)})
    return pd.concat([st, pd.DataFrame([row], columns=st.columns)], ignore_index=True)


def _league_env_30d_longprior(tg, d):
    """v0.8+ recovered league environment offset.

    Preserves the documented strictly-pregame calendar-30D semantics, while
    weakly shrinking the pooled 30D league team-game mean toward the all-pregame
    long league mean with 17 pseudo team-games. P=17 was selected only from
    2024-2025 frozen simple-coefficient parity (F5 optimum 15, late optimum 17,
    combined optimum 17); 2026 was confirmatory only.
    """
    d = pd.Timestamp(d).normalize()
    pre = tg[tg.game_date < d]
    if pre.empty:
        raise RuntimeError(f"no pregame league history before {d.date()}")
    h30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
    if h30.empty:
        h30 = pre.tail(min(len(pre), 300))
    p = LEAGUE_ENV_30D_LONG_PRIOR_P
    n = float(len(h30))
    long_f5 = float(pre.runs_f5.mean())
    long_late6 = float(pre.late_6plus_pa.mean())
    f5 = float((float(h30.runs_f5.sum()) + p * long_f5) / (n + p))
    late6 = float((float(h30.late_6plus_pa.sum()) + p * long_late6) / (n + p))
    return f5, late6


def _safe_ra9(runs: float, outs: float, fallback: float) -> float:
    return float(runs * 27.0 / outs) if outs > 0 else float(fallback)


def _bp30_reliever_pregame(bp6, bprel, *, d, opponent):
    """v0.9 forensic BP30 semantic: pregame reliever-stint, calendar36, P25.5."""
    d = pd.Timestamp(d).normalize()
    opp = str(opponent)

    pre_all = bp6[bp6.game_date < d]
    pre = pre_all[pre_all.game_date.dt.year == d.year]
    if pre.empty:
        pre = pre_all
    lg_long = _safe_ra9(float(pre.runs.sum()), float(pre.outs.sum()), 4.5)
    opp_long = pre[pre.pitching_team.astype(str).eq(opp)]
    lr = float(opp_long.runs.sum())
    lo = float(opp_long.outs.sum())
    team_long = (
        (lr + BP_LONG_OUTS_P * lg_long / 27.0) * 27.0 / (lo + BP_LONG_OUTS_P)
        if lo + BP_LONG_OUTS_P > 0 else lg_long
    )

    rec = bprel[(bprel.game_date < d) & (bprel.game_date >= d - pd.Timedelta(days=BP_RECENT_WINDOW_DAYS))]
    opp_rec = rec[rec.pitching_team.astype(str).eq(opp)]
    lg_recent = _safe_ra9(float(rec.runs.sum()), float(rec.outs.sum()), lg_long)
    rr = float(opp_rec.runs.sum())
    ro = float(opp_rec.outs.sum())
    team_recent = (
        (rr + BP_RECENT_OUTS_P * team_long / 27.0) * 27.0 / (ro + BP_RECENT_OUTS_P)
        if ro + BP_RECENT_OUTS_P > 0 else team_long
    )
    return float(team_recent / lg_recent) if lg_recent else 1.0


def build_pregame_features(tg, st, bp6, bprel, *, d, game_pk, team, opponent, self_starter, is_home, park_factor_override=None):
    fake_pk = f"RECOVERED|{game_pk}|{team}"
    # recovered.build_features looks up a starter under `opponent`; injecting the
    # SELF starter under that key reproduces the recovered starter-only historical bug.
    st2 = _inject_current_starter(st, game_pk=fake_pk, d=d, pitching_team=opponent, starter=self_starter)
    row = SimpleNamespace(game_pk=fake_pk, game_date=d, team=team, opponent=opponent, is_home=int(is_home))
    f = recovered.build_features(tg, st2, bp6, row=row)

    # v0.8+: keep the documented pregame calendar-30D league environments, with
    # weak long-prior smoothing selected from frozen 2024-2025 coefficient parity.
    league_f5_30d, league_late_30d = _league_env_30d_longprior(tg, d)
    f["league_f5_30d"] = league_f5_30d
    f["league_late_30d"] = league_late_30d

    # v0.9 BP30 forensic recovery: pregame reliever-stint, calendar36, P25.5.
    f["opp_bp_ra9_30d_rel"] = _bp30_reliever_pregame(bp6, bprel, d=d, opponent=opponent)
    if park_factor_override is not None:
        f["park_factor"] = float(park_factor_override)
    return f


def score_input(payload: dict) -> dict:
    meta = json.loads(CHALLENGER_PATH.read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    target = pd.Timestamp(payload["target_date"]).normalize()
    tg, st, bp6, bprel = _load_history(target)
    games_out = []
    for g in payload.get("games", []):
        af = build_pregame_features(
            tg, st, bp6, bprel,
            d=target,
            game_pk=str(g.get("game_pk", "UNKNOWN")),
            team=str(g["away_team"]),
            opponent=str(g["home_team"]),
            self_starter=str(g["away_starter"]),
            is_home=0,
            park_factor_override=g.get("park_factor"),
        )
        hf = build_pregame_features(
            tg, st, bp6, bprel,
            d=target,
            game_pk=str(g.get("game_pk", "UNKNOWN")),
            team=str(g["home_team"]),
            opponent=str(g["away_team"]),
            self_starter=str(g["home_starter"]),
            is_home=1,
            park_factor_override=g.get("park_factor"),
        )
        pred = engine.score(model, af, hf)
        games_out.append({
            "game_pk": g.get("game_pk"),
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "away_features": af,
            "home_features": hf,
            "mechanical_mu": pred,
        })
    return {
        "status": "RESEARCH_CHALLENGER_ACTIVE",
        "model": meta["version"],
        "official_shadow_allowed": False,
        "target_date": str(target.date()),
        "known_unresolved": meta["known_unresolved"],
        "games": games_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="state/mechanical_mu_input.json")
    ap.add_argument("--out", default="state/mu_v1_2_recovered_challenger_shadow.json")
    args = ap.parse_args()
    out = score_input(json.loads((ROOT / args.input).read_text(encoding="utf-8")))
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

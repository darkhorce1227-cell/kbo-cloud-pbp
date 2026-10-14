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
    bp6 = bp_game_tables(pa, tg)["inning6plus"].copy()
    return tg, st, bp6


def _inject_current_starter(st, *, game_pk, d, pitching_team, starter):
    row = {c: 0 for c in st.columns}
    row.update({"game_pk": str(game_pk), "game_date": d, "pitching_team": str(pitching_team), "starter": str(starter)})
    return pd.concat([st, pd.DataFrame([row], columns=st.columns)], ignore_index=True)


def _league_late_6to9_30d(tg, d):
    pre30 = tg[(tg.game_date < d) & (tg.game_date >= d - pd.Timedelta(days=30))]
    if "late_6to9_pa" not in pre30.columns or pre30.empty:
        raise RuntimeError("late_6to9_pa unavailable for recovered late offset")
    return float(pre30["late_6to9_pa"].mean())


def build_pregame_features(tg, st, bp6, *, d, game_pk, team, opponent, self_starter, is_home, park_factor_override=None):
    fake_pk = f"RECOVERED|{game_pk}|{team}"
    # recovered.build_features looks up a starter under `opponent`; injecting the
    # SELF starter under that key reproduces the recovered starter-only historical bug.
    st2 = _inject_current_starter(st, game_pk=fake_pk, d=d, pitching_team=opponent, starter=self_starter)
    row = SimpleNamespace(game_pk=fake_pk, game_date=d, team=team, opponent=opponent, is_home=int(is_home))
    f = recovered.build_features(tg, st2, bp6, row=row)
    f["league_late_30d"] = _league_late_6to9_30d(tg, d)
    if park_factor_override is not None:
        f["park_factor"] = float(park_factor_override)
    return f


def score_input(payload: dict) -> dict:
    meta = json.loads(CHALLENGER_PATH.read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    target = pd.Timestamp(payload["target_date"]).normalize()
    tg, st, bp6 = _load_history(target)
    games_out = []
    for g in payload.get("games", []):
        af = build_pregame_features(tg, st, bp6, d=target, game_pk=str(g.get("game_pk", "UNKNOWN")), team=str(g["away_team"]), opponent=str(g["home_team"]), self_starter=str(g["away_starter"]), is_home=0, park_factor_override=g.get("park_factor"))
        hf = build_pregame_features(tg, st, bp6, d=target, game_pk=str(g.get("game_pk", "UNKNOWN")), team=str(g["home_team"]), opponent=str(g["away_team"]), self_starter=str(g["home_starter"]), is_home=1, park_factor_override=g.get("park_factor"))
        pred = engine.score(model, af, hf)
        games_out.append({"game_pk": g.get("game_pk"), "away_team": g["away_team"], "home_team": g["home_team"], "away_features": af, "home_features": hf, "mechanical_mu": pred})
    return {"status": "RESEARCH_CHALLENGER_ACTIVE", "model": meta["version"], "official_shadow_allowed": False, "target_date": str(target.date()), "known_unresolved": meta["known_unresolved"], "games": games_out}


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

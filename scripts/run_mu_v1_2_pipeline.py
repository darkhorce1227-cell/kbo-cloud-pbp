from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from kbo_pbp import events as eventlib


def prepare_pa_fixed(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["game_date"] = mu._to_date(x["game_date"])
    pa = eventlib.plate_appearances(x)
    pa["game_date"] = mu._to_date(pa["game_date"])
    top = pa["inning_topbot"].astype(str).str.lower().str.startswith("top")
    pa["batting_team"] = np.where(top, pa["away_team"], pa["home_team"])
    pa["pitching_team"] = np.where(top, pa["home_team"], pa["away_team"])
    for c in ("home_score", "away_score", "post_home_score", "post_away_score", "outs_when_up", "inning"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")
    pa["runs_scored"] = (
        pa["post_home_score"].fillna(pa["home_score"])
        + pa["post_away_score"].fillna(pa["away_score"])
        - pa["home_score"].fillna(0)
        - pa["away_score"].fillna(0)
    ).clip(lower=0)

    pa["_half"] = (
        pa["game_pk"].astype(str)
        + "|"
        + pa["inning"].astype("Int64").astype(str)
        + "|"
        + pa["inning_topbot"].astype(str)
    )
    pa = pa.sort_values(
        ["game_date", "game_pk", "inning", "inning_topbot", "at_bat_number"]
    ).reset_index(drop=True)
    nxt_half = pa["_half"].shift(-1)
    nxt_outs = pa["outs_when_up"].shift(-1)
    same_half = pa["_half"].eq(nxt_half)
    out_delta = np.where(
        same_half,
        (nxt_outs - pa["outs_when_up"]).clip(lower=0),
        (3 - pa["outs_when_up"]).clip(lower=0),
    )
    walkoff_like = (
        (~same_half)
        & (pa["inning"] >= 9)
        & pa["inning_topbot"].astype(str).str.lower().str.startswith("bot")
    )
    event_outs = pa["event"].isin(mu.OUT_EVENTS).astype(int)
    event_outs += pa["event"].eq("double_play").astype(int) * 2
    event_outs += pa["event"].eq("triple_play").astype(int) * 3
    out_delta = np.where(walkoff_like, np.minimum(out_delta, event_outs.to_numpy()), out_delta)
    pa["outs_recorded"] = pd.Series(out_delta, index=pa.index, dtype="float64").fillna(0).clip(lower=0, upper=3)
    return pa


mu.prepare_pa = prepare_pa_fixed


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
    if args.cmd == "audit":
        payload = mu.audit(mu.ROOT / args.out)
    else:
        payload = mu.shadow(mu.ROOT / args.input, mu.ROOT / args.out, mu.ROOT / args.audit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

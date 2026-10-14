from __future__ import annotations

import numpy as np
import pandas as pd

import build_mu_v1_2_features as base


def prepare_pa_fixed(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["game_date"] = base._to_date(x["game_date"])
    pa = base.eventlib.plate_appearances(x)
    pa["game_date"] = base._to_date(pa["game_date"])
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
    event_outs = pa["event"].isin(base.OUT_EVENTS).astype(int)
    event_outs += pa["event"].eq("double_play").astype(int) * 2
    event_outs += pa["event"].eq("triple_play").astype(int) * 3
    out_delta = np.where(walkoff_like, np.minimum(out_delta, event_outs), out_delta)
    pa["outs_recorded"] = pd.Series(
        pd.to_numeric(out_delta, errors="coerce"), index=pa.index
    ).fillna(0).clip(lower=0, upper=3)
    return pa


base.prepare_pa = prepare_pa_fixed


if __name__ == "__main__":
    base.main()

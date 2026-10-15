from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset
import audit_mu_v1_2_bp30_leakage_forensic as bpforensic
import audit_mu_v1_2_30d_longprior_refine as refine

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
CURRENT_PATH = ROOT / "state" / "mu_v1_2_recovered_final_candidate_oos.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_30d_longprior_oos.json"
P = 17.0  # selected on 2024-2025 frozen simple coefficient parity only


def mean_mu(rows: pd.DataFrame, model: dict) -> dict[str, float]:
    f5=[]; full=[]
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away=x[x.is_home==0].iloc[0].to_dict()
        home=x[x.is_home==1].iloc[0].to_dict()
        pred=engine.score(model,away,home)
        f5 += [float(pred["f5_away"]),float(pred["f5_home"])]
        full += [float(pred["full_away"]),float(pred["full_home"])]
    return {"f5":float(np.mean(f5)),"full":float(np.mean(full))}


def map_eval(rows: pd.DataFrame, model: dict) -> dict:
    e=engine.evaluate(rows,model)
    mm=mean_mu(rows,model)
    return {
        "f5":{
            "mae":float(e["f5"]["team_score_mae"]),
            "nb_nll":float(e["f5"]["nb_nll"]),
            "wdl_brier":float(e["f5"]["wdl_brier_mean3"]),
            "wdl_logloss":float(e["f5"]["wdl_logloss"]),
            "decided_side":float(e["f5"]["decided_side_hit"]),
            "mean_mu":mm["f5"],
        },
        "full":{
            "mae":float(e["full"]["team_score_mae"]),
            "nb_nll":float(e["full"]["nb_nll"]),
            "wdl_brier":float(e["full"]["wdl_brier_mean3"]),
            "wdl_logloss":float(e["full"]["wdl_logloss"]),
            "decided_side":float(e["full"]["decided_side_hit"]),
            "mean_mu":mm["full"],
        },
    }


def main():
    model=json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    pub=json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    tg,st,bps=bpforensic.load_all()
    bp6=bps["inning6plus"]
    bprel=bps["reliever_stint"]

    rows=recovered.make_oos_rows(tg,st,bp6)
    rows=lateaudit.swap_starter(rows)
    rows=lateoffset.apply_offset_only(rows,tg,"late_6plus_pa")
    rows["opp_bp_ra9_30d_rel"]=bpforensic.candidate_bp30(
        rows,tg,bp6,bprel,leak=False,p_recent=39.25,prior_mode="abs6"
    )
    if len(rows)!=1368 or rows.game_pk.nunique()!=684:
        raise RuntimeError(f"OOS gate failed {len(rows)}/{rows.game_pk.nunique()}")

    # Change ONLY the two league environment offset columns. Offense 30D features
    # remain as in v0.7 because the historical hypothesis being tested is an
    # offset-environment smoothing rule, not a redefinition of offense features.
    dates=pd.to_datetime(rows.game_date).dt.normalize().tolist()
    stats=refine.precompute(tg,dates)
    f5off,lateoff=refine.vectors(dates,stats,P)
    rows["league_f5_30d"]=f5off
    rows["league_late_30d"]=lateoff

    got=map_eval(rows,model)
    current=(json.loads(CURRENT_PATH.read_text(encoding="utf-8"))["recovered_2026_oos"]
             if CURRENT_PATH.exists() else None)
    published={
        "f5":{**pub["oos_2026"]["f5"]["hybrid"],"decided_side":pub["decided_side_accuracy_2026"]["f5"]},
        "full":{**pub["oos_2026"]["full"]["hybrid"],"decided_side":pub["decided_side_accuracy_2026"]["full"]},
    }

    def delta(a,b):
        keys=["mae","nb_nll","wdl_brier","wdl_logloss","decided_side","mean_mu"]
        return {k:float(a[k]-b[k]) for k in keys if k in a and k in b}

    payload={
        "version":"mu-v1.2-30d-longprior-oos-v0.1",
        "status":"DIAGNOSTIC_ONLY",
        "official_shadow_allowed":False,
        "selection_discipline":"P=17 selected using 2024-2025 frozen coefficient parity only; 2026 OOS is confirmatory and cannot retune P.",
        "offset_formula":"pregame calendar30 pooled league team-game mean shrunk toward all-pregame-history league mean with 17 pseudo team-games; offseason empty-30D fallback uses most recent 300 pregame team-rows",
        "oos_team_games":int(len(rows)),
        "oos_games":int(rows.game_pk.nunique()),
        "candidate_p17":got,
        "current_v0_7":current,
        "published_v1_2":published,
        "delta_p17_minus_current":({k:delta(got[k],current[k]) for k in ["f5","full"]} if current else None),
        "delta_p17_minus_published":{k:delta(got[k],published[k]) for k in ["f5","full"]},
        "promotion_note":"This can support a v0.8 research-challenger preprocessing update only; it cannot activate Official Mechanical Shadow."
    }
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

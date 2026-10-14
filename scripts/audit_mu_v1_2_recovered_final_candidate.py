from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_recovered_final_candidate_oos.json"


def mean_mu(rows, model):
    f5=[]; full=[]
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away=x[x.is_home==0].iloc[0].to_dict()
        home=x[x.is_home==1].iloc[0].to_dict()
        p=engine.score(model,away,home)
        f5 += [float(p["f5_away"]),float(p["f5_home"])]
        full += [float(p["full_away"]),float(p["full_home"])]
    return {"f5":float(np.mean(f5)),"full":float(np.mean(full))}


def main():
    model=json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    pub=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tg,st,bp6=recovered.load_tables()

    # Historical-parity recovered semantics:
    # off_late_* remains Full-F5 history, while frozen coefficient parity strongly
    # identifies inning 6+ PA runs for the league_late_30d offset/training target.
    rows=recovered.make_oos_rows(tg,st,bp6)
    rows=lateaudit.swap_starter(rows)
    rows=lateoffset.apply_offset_only(rows,tg,"late_6plus_pa")

    if len(rows)!=1368 or rows.game_pk.nunique()!=684:
        raise RuntimeError(f"OOS gate failed rows={len(rows)} games={rows.game_pk.nunique()}")

    e=engine.evaluate(rows,model)
    mm=mean_mu(rows,model)
    got={
        "f5":{
            "mae":e["f5"]["team_score_mae"],
            "nb_nll":e["f5"]["nb_nll"],
            "wdl_brier":e["f5"]["wdl_brier_mean3"],
            "wdl_logloss":e["f5"]["wdl_logloss"],
            "decided_side":e["f5"]["decided_side_hit"],
            "mean_mu":mm["f5"],
        },
        "full":{
            "mae":e["full"]["team_score_mae"],
            "nb_nll":e["full"]["nb_nll"],
            "wdl_brier":e["full"]["wdl_brier_mean3"],
            "wdl_logloss":e["full"]["wdl_logloss"],
            "decided_side":e["full"]["decided_side_hit"],
            "mean_mu":mm["full"],
        },
    }
    published={
        "f5":{**pub["oos_2026"]["f5"]["hybrid"],"decided_side":pub["decided_side_accuracy_2026"]["f5"]},
        "full":{**pub["oos_2026"]["full"]["hybrid"],"decided_side":pub["decided_side_accuracy_2026"]["full"]},
    }
    deltas={}
    for kind in ["f5","full"]:
        p=published[kind]
        deltas[kind]={
            "mae":got[kind]["mae"]-float(p["mae"]),
            "nb_nll":got[kind]["nb_nll"]-float(p["nb_nll"]),
            "wdl_brier":got[kind]["wdl_brier"]-float(p["wdl_brier"]),
            "wdl_logloss":got[kind]["wdl_logloss"]-float(p["wdl_logloss"]),
            "mean_mu":got[kind]["mean_mu"]-float(p["mean_mu"]),
            "decided_side":got[kind]["decided_side"]-float(p["decided_side"]),
        }

    payload={
        "version":"mu-v1.2-recovered-final-candidate-v0.6-historical-parity",
        "status":"CHALLENGER_DIAGNOSTIC",
        "official_shadow_allowed":False,
        "oos_team_games":int(len(rows)),
        "oos_games":int(rows.game_pk.nunique()),
        "recovered_semantics":{
            "starter_mapping":"SELF_TEAM_STARTER",
            "late_offense_features":"FULL_MINUS_F5",
            "historical_late_training_target":"INNINGS_6_PLUS_PA_RUNS",
            "league_late_30d_offset":"INNINGS_6_PLUS_PA_RUNS",
            "bullpen_mapping":"OPPONENT_BULLPEN",
            "starter_rate_prior":"league->long 150 BF; long->recent5 150 BF",
            "starter_ip_prior":"league->long 5 starts; long->recent5 5 starts",
        },
        "recovered_2026_oos":got,
        "published_v1_2_hybrid_2026_oos":published,
        "delta_recovered_minus_published":deltas,
        "note":"Diagnostic only. Frozen v1.2 coefficients/scalers/gamma/NB alpha are unchanged. Historical late semantics are chosen from 2024-2025 frozen coefficient parity, not 2026 OOS selection. Official Shadow remains gated on BP30 and exact row parity.",
    }
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

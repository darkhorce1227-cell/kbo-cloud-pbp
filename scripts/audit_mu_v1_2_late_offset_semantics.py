from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_late_offset_semantics.json"


def apply_offset_only(rows, tg, target_col):
    z=rows.copy(deep=True)
    vals=[]
    for r in z.itertuples(index=False):
        d=pd.Timestamp(r.game_date).normalize()
        pre=tg[(tg.game_date<d)&(tg.game_date>=recovered.HISTORY_START)]
        pre30=pre[pre.game_date>=d-pd.Timedelta(days=30)]
        lg=float(pre[target_col].mean()) if len(pre) else 0.0
        lg30=float(pre30[target_col].mean()) if len(pre30) else lg
        vals.append(lg30)
    z["league_late_30d"]=vals
    return z


def mean_mu(rows, model):
    full=[]
    for _,x in rows.groupby("game_pk"):
        if len(x)!=2:
            continue
        away=x[x.is_home==0].iloc[0].to_dict(); home=x[x.is_home==1].iloc[0].to_dict()
        p=engine.score(model,away,home)
        full += [float(p["full_away"]),float(p["full_home"])]
    return float(np.mean(full))


def summarize(rows,model):
    e=engine.evaluate(rows,model)
    return {
        "mae":e["full"]["team_score_mae"],
        "nb_nll":e["full"]["nb_nll"],
        "wdl_brier":e["full"]["wdl_brier_mean3"],
        "wdl_logloss":e["full"]["wdl_logloss"],
        "side":e["full"]["decided_side_hit"],
        "mean_mu":mean_mu(rows,model),
    }


def main():
    model=json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    pub=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tg,st,bp6=recovered.load_tables()
    base=recovered.make_oos_rows(tg,st,bp6)
    base=lateaudit.swap_starter(base)
    if len(base)!=1368 or base.game_pk.nunique()!=684:
        raise RuntimeError(f"OOS gate failed rows={len(base)} games={base.game_pk.nunique()}")

    variants={
        "features_fullminus_offset_fullminus":base,
        "features_fullminus_offset_6plus":apply_offset_only(base,tg,"late_6plus_pa"),
        "features_fullminus_offset_6to9":apply_offset_only(base,tg,"late_6to9_pa"),
        "features_6to9_offset_6to9":lateaudit.apply_late_features(base,tg,"late_6to9_pa"),
    }
    published={
        **pub["oos_2026"]["full"]["hybrid"],
        "side":pub["decided_side_accuracy_2026"]["full"],
    }
    ranked=[]
    for name,rows in variants.items():
        s=summarize(rows,model)
        dist=(abs(s["mean_mu"]-float(published["mean_mu"]))
              +abs(s["mae"]-float(published["mae"]))
              +abs(s["wdl_logloss"]-float(published["wdl_logloss"])))
        ranked.append({"variant":name,"distance":float(dist),**s})
    ranked.sort(key=lambda x:x["distance"])
    payload={
        "version":"mu-v1.2-late-offset-semantics-v0.1",
        "status":"DIAGNOSTIC_ONLY",
        "official_shadow_allowed":False,
        "rows":1368,
        "games":684,
        "starter_mapping":"SELF_TEAM_STARTER",
        "published_full":published,
        "ranked":ranked,
        "note":"Tests late offense feature semantics separately from the league_late_30d offset semantics. No production promotion is authorized by this file alone.",
    }
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

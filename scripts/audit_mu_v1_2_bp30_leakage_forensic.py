from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as base
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"research"/"mu-v1.2"/"model_spec.json"
OUT=ROOT/"state"/"mu_v1_2_bp30_leakage_forensic.json"
P_LONG=360.0


def load_all():
    base.prepare_pa=runner.prepare_pa_fixed
    raw=base.load_seasons(2026)
    pa=runner.prepare_pa_fixed(raw)
    tg,st_base,_=make_team_game(pa)
    st=starter_game_table(pa).copy()
    ip_map={(str(r.game_pk),str(r.pitching_team)):float(r.ip) for r in st_base.itertuples(index=False)}
    st["ip"]=[ip_map.get((str(r.game_pk),str(r.pitching_team)),np.nan) for r in st.itertuples(index=False)]
    if st["ip"].isna().any(): raise RuntimeError("starter IP merge missing")
    bps=bp_game_tables(pa,tg)
    return tg,st,bps

def ra9(r,o,f=4.5): return float(r*27.0/o) if o>0 else float(f)

def recent_slice(z,d,gpk,leak):
    if leak:
        x=z[(z.game_date<=d)&(z.game_date>=d-pd.Timedelta(days=30))]
        return x[(x.game_date<d)|(x.game_pk.astype(str)==gpk)]
    return z[(z.game_date<d)&(z.game_date>=d-pd.Timedelta(days=30))]

def candidate_bp30(rows,tg,bp6,bprel,*,leak,p_recent,prior_mode):
    vals=[]
    for r in rows.itertuples(index=False):
        d=pd.Timestamp(r.game_date).normalize();opp=str(r.opponent);gpk=str(r.game_pk)
        preall=bp6[bp6.game_date<d];pre=preall[preall.game_date.dt.year==d.year]
        if pre.empty:pre=preall
        lg6=ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        op=pre[pre.pitching_team.astype(str).eq(opp)]
        rr=float(op.runs.sum());oo=float(op.outs.sum())
        team6=(rr+P_LONG*lg6/27.0)*27.0/(oo+P_LONG) if oo+P_LONG>0 else lg6
        longrel6=team6/lg6 if lg6 else 1.0

        rec=recent_slice(bprel,d,gpk,leak)
        oppr=rec[rec.pitching_team.astype(str).eq(opp)]
        den=ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg6)
        rrun=float(oppr.runs.sum()); rout=float(oppr.outs.sum())
        if prior_mode=="abs6":
            prior=team6
        elif prior_mode=="longrel_x_six30":
            six30=recent_slice(bp6,d,gpk,leak)
            lg6_30=ra9(float(six30.runs.sum()),float(six30.outs.sum()),lg6)
            prior=longrel6*lg6_30
        else:
            raise ValueError(prior_mode)
        num=(rrun+p_recent*prior/27.0)*27.0/(rout+p_recent) if rout+p_recent>0 else prior
        vals.append(num/den if den else 1.0)
    return np.asarray(vals,float)

def mean_mu(rows,model):
    f5=[];full=[]
    for _,x in rows.groupby("game_pk"):
        if len(x)!=2:continue
        a=x[x.is_home==0].iloc[0].to_dict();h=x[x.is_home==1].iloc[0].to_dict()
        p=engine.score(model,a,h)
        f5 += [float(p["f5_away"]),float(p["f5_home"])]
        full += [float(p["full_away"]),float(p["full_home"])]
    return {"f5":float(np.mean(f5)),"full":float(np.mean(full))}

def oos_metrics(rows,model):
    e=engine.evaluate(rows,model); mm=mean_mu(rows,model)
    return {
        "f5":{"mae":e["f5"]["team_score_mae"],"logloss":e["f5"]["wdl_logloss"],"side":e["f5"]["decided_side_hit"],"mean_mu":mm["f5"]},
        "full":{"mae":e["full"]["team_score_mae"],"logloss":e["full"]["wdl_logloss"],"side":e["full"]["decided_side_hit"],"mean_mu":mm["full"]},
    }

def main():
    model=json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    tg,st,bps=load_all();bp6=bps["inning6plus"];bprel=bps["reliever_stint"]

    dev=coeff.make_dev_rows(tg,st,bp6)
    dev=lateoffset.apply_offset_only(dev,tg,"late_6plus_pa")
    dev=lateaudit.swap_starter(dev)
    oos=recovered.make_oos_rows(tg,st,bp6)
    oos=lateoffset.apply_offset_only(oos,tg,"late_6plus_pa")
    oos=lateaudit.swap_starter(oos)
    if len(oos)!=1368 or oos.game_pk.nunique()!=684: raise RuntimeError("OOS gate")

    variants=[
      {"name":"BASE_V06_INNING6PLUS_P54","mode":"base"},
      {"name":"RELIEVER_PREGAME_ABS6_P39_25","mode":"rel","leak":False,"p":39.25,"prior":"abs6"},
      {"name":"RELIEVER_SAMEGAME_ABS6_P35","mode":"rel","leak":True,"p":35.0,"prior":"abs6"},
      {"name":"RELIEVER_SAMEGAME_RELPRIOR_P34_25","mode":"rel","leak":True,"p":34.25,"prior":"longrel_x_six30"},
    ]
    results=[]
    for v in variants:
        d=dev.copy();o=oos.copy()
        if v["mode"]=="rel":
            d["opp_bp_ra9_30d_rel"]=candidate_bp30(d,tg,bp6,bprel,leak=v["leak"],p_recent=v["p"],prior_mode=v["prior"])
            o["opp_bp_ra9_30d_rel"]=candidate_bp30(o,tg,bp6,bprel,leak=v["leak"],p_recent=v["p"],prior_mode=v["prior"])
        cf5=coeff.fit_component(d,model,"core_f5",engine.CORE_F5,"runs_f5","league_f5_30d",5.0)
        cl=coeff.fit_component(d,model,"core_late",engine.CORE_LATE,"runs_late_6plus","league_late_30d",300.0)
        moment={"mean":float(d["opp_bp_ra9_30d_rel"].mean()),"scale":float(d["opp_bp_ra9_30d_rel"].std(ddof=0))}
        results.append({
          "variant":v["name"],"leakage":bool(v.get("leak",False)),"bp30_moment":moment,
          "coefficient_parity":{"core_f5_norm":cf5["param_norm_mae"],"core_f5_corr":cf5["param_corr"],"core_late_norm":cl["param_norm_mae"],"core_late_corr":cl["param_corr"],"core_sum":float(cf5["param_norm_mae"]+cl["param_norm_mae"])},
          "oos_2026":oos_metrics(o,model),
        })
    results.sort(key=lambda x:x["coefficient_parity"]["core_sum"])
    payload={
      "version":"mu-v1.2-bp30-leakage-forensic-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,
      "frozen_bp30":{"mean":0.9860898056018214,"scale":0.21158719158114866},
      "published_oos":{"f5":{"mae":1.930994,"logloss":0.993735,"side":0.560272,"mean_mu":2.668074},"full":{"mae":2.705128,"logloss":0.813524,"side":0.571856,"mean_mu":5.001728}},
      "results":results,
      "decision_rule":"Do not infer leakage from moment fit alone. A leakage candidate must also improve frozen coefficient parity and 2026 OOS reproduction materially before it can be treated as historical implementation evidence."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

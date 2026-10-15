from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT=Path(__file__).resolve().parents[1]
SHA24="d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25="017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"
P_LONG=360.0
FEAT="opp_bp_ra9_30d_rel"
OUT=ROOT/"state"/"mu_v1_2_bp30_recent_source.json"


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"]).reset_index(drop=True)

def ra9(r,o,f=4.5): return float(r*27.0/o) if o>0 else float(f)

def moments(a):
    a=np.asarray(a,float); fm=float(FROZEN[FEAT]["mean"]); fs=float(FROZEN[FEAT]["scale"])
    m=float(np.mean(a)); s=float(np.std(a,ddof=0))
    return {"mean":m,"scale":s,"mean_error":m-fm,"scale_error":s-fs,"frozen_mean":fm,"frozen_scale":fs,"loss":abs(m-fm)/fs+abs(s-fs)/fs}

def load():
    td=ROOT/"out_training"; p24=td/"kbo_pbp_2024_training.csv.gz";p25=td/"kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    raw=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(raw); tg,_,_=make_team_game(pa); bps=bp_game_tables(pa,tg)
    return tg,bps

def equal_team_raw(z, fallback):
    g=z.groupby("pitching_team",as_index=False).agg(runs=("runs","sum"),outs=("outs","sum"))
    a=[ra9(float(r.runs),float(r.outs),np.nan) for r in g.itertuples(index=False) if float(r.outs)>0]
    return float(np.mean(a)) if a else float(fallback)

def build_one(tg,long_bp,recent_bp,include_current,denom_mode,p_recent):
    out=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent); gpk=str(r.game_pk)
        pre_all=long_bp[long_bp.game_date<d]; pre=pre_all[pre_all.game_date.dt.year==d.year]
        if pre.empty: pre=pre_all
        oppl=pre[pre.pitching_team.astype(str).eq(opp)]
        lg_long=ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        tr=float(oppl.runs.sum());to=float(oppl.outs.sum())
        team_long=(tr+P_LONG*lg_long/27.0)*27.0/(to+P_LONG) if to+P_LONG>0 else lg_long
        if include_current:
            rec=recent_bp[(recent_bp.game_date<=d)&(recent_bp.game_date>=d-pd.Timedelta(days=30))]
            # On the current date, retain only the current game's rows, avoiding other same-day future games.
            rec=rec[(rec.game_date<d)|(rec.game_pk.astype(str)==gpk)]
        else:
            rec=recent_bp[(recent_bp.game_date<d)&(recent_bp.game_date>=d-pd.Timedelta(days=30))]
        opprec=rec[rec.pitching_team.astype(str).eq(opp)]
        pooled=ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg_long)
        den=pooled if denom_mode=="pooled" else equal_team_raw(rec,pooled)
        rr=float(opprec.runs.sum());ro=float(opprec.outs.sum())
        num=(rr+p_recent*team_long/27.0)*27.0/(ro+p_recent) if ro+p_recent>0 else team_long
        out.append(num/den if den else 1.0)
    return np.asarray(out,float)

def main():
    tg,bps=load(); assert len(dev(tg))==2628
    candidates=[]
    for source in ["inning6plus","reliever_stint","inning6to9"]:
      for include_current in [False,True]:
       for denom in ["pooled","equal_team_raw"]:
        for p in np.arange(0,121,3,dtype=float):
            a=build_one(tg,bps["inning6plus"],bps[source],include_current,denom,float(p))
            m=moments(a)
            candidates.append({"recent_source":source,"include_current_game":include_current,"denominator":denom,"p_recent_outs":float(p),**m})
    candidates.sort(key=lambda x:x["loss"])
    payload={"version":"mu-v1.2-bp30-recent-source-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,"rows":2628,"top":candidates[:30],"best_by_family":{},"note":"Tests recent bullpen source, same-game inclusion leakage, denominator aggregation, and recent pseudo-outs. Long BP remains pregame inning6plus with 360-out prior."}
    for x in candidates:
        key=f"{x['recent_source']}|include_current={x['include_current_game']}|{x['denominator']}"
        if key not in payload["best_by_family"]: payload["best_by_family"][key]=x
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

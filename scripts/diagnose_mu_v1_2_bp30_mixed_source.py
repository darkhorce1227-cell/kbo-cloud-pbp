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
OUT=ROOT/"state"/"mu_v1_2_bp30_mixed_source.json"
SOURCES=("inning6plus","reliever_stint")


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

def slice_recent(z,d,gpk,include_current):
    if include_current:
        x=z[(z.game_date<=d)&(z.game_date>=d-pd.Timedelta(days=30))]
        return x[(x.game_date<d)|(x.game_pk.astype(str)==gpk)]
    return z[(z.game_date<d)&(z.game_date>=d-pd.Timedelta(days=30))]

def family_stats(tg,bps,run_source,out_source,denom_source,include_current):
    rr=[]; ro=[]; tl=[]; den=[]
    long_bp=bps["inning6plus"]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent); gpk=str(r.game_pk)
        pre_all=long_bp[long_bp.game_date<d]; pre=pre_all[pre_all.game_date.dt.year==d.year]
        if pre.empty: pre=pre_all
        oppl=pre[pre.pitching_team.astype(str).eq(opp)]
        lg_long=ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        tr=float(oppl.runs.sum()); to=float(oppl.outs.sum())
        team_long=(tr+P_LONG*lg_long/27.0)*27.0/(to+P_LONG) if to+P_LONG>0 else lg_long

        zr=slice_recent(bps[run_source],d,gpk,include_current)
        zo=slice_recent(bps[out_source],d,gpk,include_current)
        zd=slice_recent(bps[denom_source],d,gpk,include_current)
        rr.append(float(zr.loc[zr.pitching_team.astype(str).eq(opp),"runs"].sum()))
        ro.append(float(zo.loc[zo.pitching_team.astype(str).eq(opp),"outs"].sum()))
        den.append(ra9(float(zd.runs.sum()),float(zd.outs.sum()),lg_long))
        tl.append(team_long)
    return np.asarray(rr),np.asarray(ro),np.asarray(tl),np.asarray(den)

def eval_grid(rr,ro,tl,den):
    ans=[]
    for p in np.arange(10.0,70.01,0.5):
        d=ro+p
        num=np.where(d>0,(rr+p*tl/27.0)*27.0/d,tl)
        rel=np.where(den!=0,num/den,1.0)
        ans.append((float(p),moments(rel)))
    return ans

def main():
    tg,bps=load(); assert len(dev(tg))==2628
    cand=[]
    for rs in SOURCES:
      for os in SOURCES:
       for ds in SOURCES:
        for leak in (False,True):
            rr,ro,tl,dn=family_stats(tg,bps,rs,os,ds,leak)
            for p,m in eval_grid(rr,ro,tl,dn):
                cand.append({"run_source":rs,"out_source":os,"league_denominator_source":ds,"include_current_game":leak,"p_recent_outs":p,**m})
    cand.sort(key=lambda x:x["loss"])
    bestfam={}
    for x in cand:
        k=f"runs={x['run_source']}|outs={x['out_source']}|den={x['league_denominator_source']}|current={x['include_current_game']}"
        bestfam.setdefault(k,x)
    payload={"version":"mu-v1.2-bp30-mixed-source-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,"rows":2628,"top":cand[:40],"best_by_family":bestfam,"decision_note":"No arbitrary multiplicative calibration. Candidate semantics vary only mechanically plausible recent runs, outs, league denominator source, current-game inclusion, and pseudo-outs; long feature remains inning6plus pregame P360."}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

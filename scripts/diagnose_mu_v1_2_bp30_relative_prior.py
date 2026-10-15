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
OUT=ROOT/"state"/"mu_v1_2_bp30_relative_prior.json"


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"]).reset_index(drop=True)

def ra9(r,o,f=4.5): return float(r*27.0/o) if o>0 else float(f)

def moments(a):
    a=np.asarray(a,float); fm=float(FROZEN[FEAT]["mean"]); fs=float(FROZEN[FEAT]["scale"])
    m=float(np.mean(a)); s=float(np.std(a,ddof=0))
    return {"mean":m,"scale":s,"mean_error":m-fm,"scale_error":s-fs,"frozen_mean":fm,"frozen_scale":fs,"loss":abs(m-fm)/fs+abs(s-fs)/fs}

def load():
    td=ROOT/"out_training";p24=td/"kbo_pbp_2024_training.csv.gz";p25=td/"kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    raw=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(raw);tg,_,_=make_team_game(pa);return tg,bp_game_tables(pa,tg)

def recslice(z,d,gpk,leak):
    if leak:
        x=z[(z.game_date<=d)&(z.game_date>=d-pd.Timedelta(days=30))]
        return x[(x.game_date<d)|(x.game_pk.astype(str)==gpk)]
    return z[(z.game_date<d)&(z.game_date>=d-pd.Timedelta(days=30))]

def cache(tg,bps):
    six=bps["inning6plus"]; rel=bps["reliever_stint"]
    out={leak:{k:[] for k in ("rr","ro","den_rel30","prior_abs6","prior_rel_x_rel_long","prior_rel_x_rel30","prior_rel_x_six30")} for leak in (False,True)}
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date);opp=str(r.opponent);gpk=str(r.game_pk)
        pre6all=six[six.game_date<d];pre6=pre6all[pre6all.game_date.dt.year==d.year]
        if pre6.empty:pre6=pre6all
        lg6=ra9(float(pre6.runs.sum()),float(pre6.outs.sum()),4.5)
        op6=pre6[pre6.pitching_team.astype(str).eq(opp)]
        r6=float(op6.runs.sum());o6=float(op6.outs.sum())
        team6=(r6+P_LONG*lg6/27.0)*27.0/(o6+P_LONG) if o6+P_LONG>0 else lg6
        long_rel6=team6/lg6 if lg6 else 1.0

        prerall=rel[rel.game_date<d];prer=prerall[prerall.game_date.dt.year==d.year]
        if prer.empty:prer=prerall
        lg_rel_long=ra9(float(prer.runs.sum()),float(prer.outs.sum()),4.5)

        for leak in (False,True):
            rrw=recslice(rel,d,gpk,leak); oppr=rrw[rrw.pitching_team.astype(str).eq(opp)]
            den_rel30=ra9(float(rrw.runs.sum()),float(rrw.outs.sum()),lg_rel_long)
            sixw=recslice(six,d,gpk,leak)
            den_six30=ra9(float(sixw.runs.sum()),float(sixw.outs.sum()),lg6)
            c=out[leak]
            c["rr"].append(float(oppr.runs.sum()));c["ro"].append(float(oppr.outs.sum()));c["den_rel30"].append(den_rel30)
            c["prior_abs6"].append(team6)
            c["prior_rel_x_rel_long"].append(long_rel6*lg_rel_long)
            c["prior_rel_x_rel30"].append(long_rel6*den_rel30)
            c["prior_rel_x_six30"].append(long_rel6*den_six30)
    for c in out.values():
        for k in c:c[k]=np.asarray(c[k],float)
    return out

def main():
    tg,bps=load(); assert len(dev(tg))==2628
    C=cache(tg,bps);cand=[]
    modes=("prior_abs6","prior_rel_x_rel_long","prior_rel_x_rel30","prior_rel_x_six30")
    for leak in (False,True):
      c=C[leak]
      for mode in modes:
       prior=c[mode]
       for p in np.arange(10.0,70.001,0.25):
        dd=c["ro"]+p
        num=np.where(dd>0,(c["rr"]+p*prior/27.0)*27.0/dd,prior)
        rel=np.where(c["den_rel30"]!=0,num/c["den_rel30"],1.0)
        cand.append({"include_current_game":leak,"prior_mode":mode,"p_recent_outs":float(p),**moments(rel)})
    cand.sort(key=lambda x:x["loss"])
    fam={}
    for x in cand:fam.setdefault(f"{x['prior_mode']}|current={x['include_current_game']}",x)
    payload={"version":"mu-v1.2-bp30-relative-prior-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,"rows":2628,"fixed":"recent numerator and denominator are reliever_stint; exported long BP feature remains inning6plus P360","top":cand[:40],"best_by_family":fam,"note":"Tests mechanically plausible hierarchical relative priors: long_rel6 multiplied by reliever long/recent league RA9 or inning6plus recent league RA9. No arbitrary scalar calibration."}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

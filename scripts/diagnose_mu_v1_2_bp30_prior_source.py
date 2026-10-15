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
OUT=ROOT/"state"/"mu_v1_2_bp30_prior_source.json"


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"]).reset_index(drop=True)

def ra9(r,o,f=4.5): return float(r*27.0/o) if o>0 else float(f)

def moments(a):
    a=np.asarray(a,float); fm=float(FROZEN[FEAT]["mean"]); fs=float(FROZEN[FEAT]["scale"])
    m=float(np.mean(a)); s=float(np.std(a,ddof=0))
    return {"mean":m,"scale":s,"mean_error":m-fm,"scale_error":s-fs,"frozen_mean":fm,"frozen_scale":fs,"loss":abs(m-fm)/fs+abs(s-fs)/fs}

def load():
    td=ROOT/"out_training"; p24=td/"kbo_pbp_2024_training.csv.gz"; p25=td/"kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    raw=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(raw); tg,_,_=make_team_game(pa); return tg,bp_game_tables(pa,tg)

def recslice(z,d,gpk,leak):
    if leak:
        x=z[(z.game_date<=d)&(z.game_date>=d-pd.Timedelta(days=30))]
        return x[(x.game_date<d)|(x.game_pk.astype(str)==gpk)]
    return z[(z.game_date<d)&(z.game_date>=d-pd.Timedelta(days=30))]

def cache(tg,bps):
    six=bps["inning6plus"]; rel=bps["reliever_stint"]
    rows=dev(tg)
    out={leak:{"rr":[],"ro":[],"den":[],"prior_six":[],"prior_rel":[],"prior_rel_raw":[],"prior_lg_rel":[]} for leak in (False,True)}
    for r in rows.itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent); gpk=str(r.game_pk)
        # inning6+ season-reset long prior
        pre6all=six[six.game_date<d]; pre6=pre6all[pre6all.game_date.dt.year==d.year]
        if pre6.empty: pre6=pre6all
        lg6=ra9(float(pre6.runs.sum()),float(pre6.outs.sum()),4.5)
        o6=pre6[pre6.pitching_team.astype(str).eq(opp)]
        r6=float(o6.runs.sum()); q6=float(o6.outs.sum())
        team6=(r6+P_LONG*lg6/27.0)*27.0/(q6+P_LONG) if q6+P_LONG>0 else lg6

        # reliever-stint season-reset long prior
        prerall=rel[rel.game_date<d]; prer=prerall[prerall.game_date.dt.year==d.year]
        if prer.empty: prer=prerall
        lgr=ra9(float(prer.runs.sum()),float(prer.outs.sum()),4.5)
        orr=prer[prer.pitching_team.astype(str).eq(opp)]
        rlr=float(orr.runs.sum()); rlo=float(orr.outs.sum())
        teamr=(rlr+P_LONG*lgr/27.0)*27.0/(rlo+P_LONG) if rlo+P_LONG>0 else lgr
        teamr_raw=ra9(rlr,rlo,lgr)

        for leak in (False,True):
            rec=recslice(rel,d,gpk,leak); oppr=rec[rec.pitching_team.astype(str).eq(opp)]
            c=out[leak]
            c["rr"].append(float(oppr.runs.sum())); c["ro"].append(float(oppr.outs.sum()))
            c["den"].append(ra9(float(rec.runs.sum()),float(rec.outs.sum()),lgr))
            c["prior_six"].append(team6); c["prior_rel"].append(teamr)
            c["prior_rel_raw"].append(teamr_raw); c["prior_lg_rel"].append(lgr)
    for c in out.values():
        for k in c: c[k]=np.asarray(c[k],float)
    return out

def main():
    tg,bps=load(); assert len(dev(tg))==2628
    C=cache(tg,bps); cand=[]
    modes=("prior_six","prior_rel","prior_rel_raw","prior_lg_rel")
    for leak in (False,True):
      c=C[leak]
      for mode in modes:
       prior=c[mode]
       for p in np.arange(10.0,70.001,0.25):
        denouts=c["ro"]+p
        num=np.where(denouts>0,(c["rr"]+p*prior/27.0)*27.0/denouts,prior)
        rel=np.where(c["den"]!=0,num/c["den"],1.0)
        cand.append({"include_current_game":leak,"prior_mode":mode,"p_recent_outs":float(p),**moments(rel)})
    cand.sort(key=lambda x:x["loss"])
    fam={}
    for x in cand:
        fam.setdefault(f"{x['prior_mode']}|current={x['include_current_game']}",x)
    payload={"version":"mu-v1.2-bp30-prior-source-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,"rows":2628,"fixed":"recent runs/outs/league denominator = reliever_stint; long exported BP feature remains inning6plus","top":cand[:40],"best_by_family":fam,"note":"P_LONG fixed at 360; no arbitrary multiplicative calibration. Tests whether the hierarchical prior feeding BP30 came from inning6plus long, reliever-stint long, raw reliever long, or league reliever long."}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

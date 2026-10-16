from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import audit_mu_v1_2_bp30_leakage_forensic as src

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state'/'mu_v1_2_bp30_window_boundary.json'
FM=0.9860898056018214
FS=0.21158719158114866
P_LONG=360.0
TRAIN_START=pd.Timestamp('2024-05-01')


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(['game_date','game_pk','is_home']).reset_index(drop=True)


def ra9(r,o,f=4.5):
    return float(r*27.0/o) if o>0 else float(f)


def stats_for_window(tg,long_bp,rel_bp,days):
    rr=[]; ro=[]; tl=[]; den=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent)
        pre_all=long_bp[long_bp.game_date<d]
        pre=pre_all[pre_all.game_date.dt.year==d.year]
        if pre.empty: pre=pre_all
        oppl=pre[pre.pitching_team.astype(str).eq(opp)]
        lg_long=ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        tr=float(oppl.runs.sum()); to=float(oppl.outs.sum())
        team_long=(tr+P_LONG*lg_long/27.0)*27.0/(to+P_LONG) if to+P_LONG>0 else lg_long
        rec=rel_bp[(rel_bp.game_date<d)&(rel_bp.game_date>=d-pd.Timedelta(days=int(days)))]
        opprec=rec[rec.pitching_team.astype(str).eq(opp)]
        rr.append(float(opprec.runs.sum())); ro.append(float(opprec.outs.sum())); tl.append(team_long)
        den.append(ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg_long))
    return tuple(np.asarray(x,float) for x in (rr,ro,tl,den))


def eval_one(st,p):
    rr,ro,tl,den=st
    x=ro+p
    num=np.where(x>0,(rr+p*tl/27.0)*27.0/x,tl)
    rel=np.where(np.abs(den)>1e-12,num/den,1.0)
    m=float(rel.mean()); s=float(rel.std(ddof=0))
    dist=float(np.sqrt(((m-FM)/FS)**2+((s-FS)/FS)**2))
    return m,s,dist


def main():
    tg,_,bps=src.load_all()
    assert len(dev(tg))==2628
    rows=[]
    for days in range(27,34):
        st=stats_for_window(tg,bps['inning6plus'],bps['reliever_stint'],days)
        for p in np.arange(20.0,50.01,0.5):
            m,s,dist=eval_one(st,float(p))
            rows.append({'window_days':days,'p_recent_outs':float(p),'mean':m,'scale':s,'mean_error':m-FM,'scale_error':s-FS,'moment_distance':dist})
    ranked=sorted(rows,key=lambda x:x['moment_distance'])
    payload={
      'version':'mu-v1.2-bp30-window-boundary-v0.2','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,
      'fixed':'reliever_stint; pregame; pooled reliever league denominator; inning6plus long prior P360',
      'scan':{'window_days':[27,28,29,30,31,32,33],'p_recent_outs':'20..50 step0.5'},
      'frozen':{'mean':FM,'scale':FS},'top':ranked[:20],
      'best_by_window':{str(d):min((x for x in rows if x['window_days']==d),key=lambda x:x['moment_distance']) for d in range(27,34)},
      'note':'No OOS metric and no coefficient fitting; this isolates only calendar-window boundary semantics.'
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

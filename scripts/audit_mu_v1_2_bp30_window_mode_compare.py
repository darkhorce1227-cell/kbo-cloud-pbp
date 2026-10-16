from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import audit_mu_v1_2_bp30_leakage_forensic as src

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state'/'mu_v1_2_bp30_window_mode_compare.json'
FM=0.9860898056018214
FS=0.21158719158114866
P_LONG=360.0
TRAIN_START=pd.Timestamp('2024-05-01')


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(['game_date','game_pk','is_home']).reset_index(drop=True)

def ra9(r,o,f=4.5): return float(r*27.0/o) if o>0 else float(f)

def row_base(long_bp,d,opp):
    preall=long_bp[long_bp.game_date<d]
    pre=preall[preall.game_date.dt.year==d.year]
    if pre.empty: pre=preall
    lg=ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
    op=pre[pre.pitching_team.astype(str).eq(opp)]
    team=(float(op.runs.sum())+P_LONG*lg/27.0)*27.0/(float(op.outs.sum())+P_LONG)
    return lg,team

def collect(tg,long_bp,rel_bp,mode):
    rr=[]; ro=[]; tl=[]; den=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent)
        lg,team=row_base(long_bp,d,opp)
        if mode=='calendar36':
            rec=rel_bp[(rel_bp.game_date<d)&(rel_bp.game_date>=d-pd.Timedelta(days=36))]
        elif mode=='last30_team_games':
            opphist=rel_bp[(rel_bp.game_date<d)&(rel_bp.pitching_team.astype(str).eq(opp))].sort_values(['game_date','game_pk'])
            last=opphist.tail(30)
            cutoff=last.game_date.min() if len(last) else d
            rec=rel_bp[(rel_bp.game_date<d)&(rel_bp.game_date>=cutoff)]
        else: raise ValueError(mode)
        o=rec[rec.pitching_team.astype(str).eq(opp)]
        rr.append(float(o.runs.sum())); ro.append(float(o.outs.sum())); tl.append(team)
        den.append(ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg))
    return tuple(np.asarray(x,float) for x in (rr,ro,tl,den))

def eval_grid(st):
    rr,ro,tl,den=st; out=[]
    for p in np.arange(20.0,45.01,0.25):
        x=ro+p; num=np.where(x>0,(rr+p*tl/27.0)*27.0/x,tl)
        rel=np.where(np.abs(den)>1e-12,num/den,1.0)
        m=float(rel.mean()); s=float(rel.std(ddof=0))
        dist=float(np.sqrt(((m-FM)/FS)**2+((s-FS)/FS)**2))
        out.append({'p_recent_outs':float(p),'mean':m,'scale':s,'mean_error':m-FM,'scale_error':s-FS,'moment_distance':dist})
    return sorted(out,key=lambda x:x['moment_distance'])

def main():
    tg,_,bps=src.load_all(); assert len(dev(tg))==2628
    by={}
    for mode in ['calendar36','last30_team_games']:
        ranked=eval_grid(collect(tg,bps['inning6plus'],bps['reliever_stint'],mode))
        by[mode]={'best':ranked[0],'top5':ranked[:5]}
    payload={'version':'mu-v1.2-bp30-window-mode-compare-v0.1','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,'fixed':'reliever_stint; pregame; pooled reliever league denominator; inning6plus long prior P360','frozen':{'mean':FM,'scale':FS},'p_scan':'20..45 step0.25','by_mode':by,'note':'No OOS metrics or coefficient fitting. Direct semantic comparison of the 36-calendar-day forensic hit against last-30-opponent-team-games.'}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

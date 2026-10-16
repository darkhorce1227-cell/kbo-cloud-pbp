from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import audit_mu_v1_2_bp30_leakage_forensic as src
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_late_offset_semantics as lateoffset
import audit_mu_v1_2_late_total_variants as lateaudit
import mu_v1_3 as engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state'/'mu_v1_2_bp30_reliever_denominator_hybrid.json'
FM=0.9860898056018214
FS=0.21158719158114866
P_LONG=360.0


def stats(rows,bp6,bprel):
    rr=[]; oo=[]; prior=[]; den_rel=[]; den_6=[]
    for r in rows.itertuples(index=False):
        d=pd.Timestamp(r.game_date).normalize(); opp=str(r.opponent)
        preall=bp6[bp6.game_date<d]; pre=preall[preall.game_date.dt.year==d.year]
        if pre.empty: pre=preall
        lg6=src.ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        op=pre[pre.pitching_team.astype(str).eq(opp)]
        team6=(float(op.runs.sum())+P_LONG*lg6/27.0)*27.0/(float(op.outs.sum())+P_LONG)
        rel30=bprel[(bprel.game_date<d)&(bprel.game_date>=d-pd.Timedelta(days=30))]
        six30=bp6[(bp6.game_date<d)&(bp6.game_date>=d-pd.Timedelta(days=30))]
        opprel=rel30[rel30.pitching_team.astype(str).eq(opp)]
        rr.append(float(opprel.runs.sum())); oo.append(float(opprel.outs.sum())); prior.append(team6)
        den_rel.append(src.ra9(float(rel30.runs.sum()),float(rel30.outs.sum()),lg6))
        den_6.append(src.ra9(float(six30.runs.sum()),float(six30.outs.sum()),lg6))
    return tuple(np.asarray(x,float) for x in (rr,oo,prior,den_rel,den_6))


def vals(st,p,den_i):
    rr,oo,prior,den_rel,den_6=st
    den=den_rel if den_i=='RELIEVER_LEAGUE30' else den_6
    num=(rr+p*prior/27.0)*27.0/(oo+p)
    return np.divide(num,den,out=np.ones_like(num),where=np.abs(den)>1e-12)


def main():
    model=json.loads(src.MODEL_PATH.read_text(encoding='utf-8'))
    tg,st0,bps=src.load_all(); bp6=bps['inning6plus']; bprel=bps['reliever_stint']
    dev=coeff.make_dev_rows(tg,st0,bp6)
    dev=lateoffset.apply_offset_only(dev,tg,'late_6plus_pa'); dev=lateaudit.swap_starter(dev)
    if len(dev)!=2628: raise RuntimeError('dev gate')
    st=stats(dev,bp6,bprel)
    rows=[]; d=dev.copy()
    for den in ['RELIEVER_LEAGUE30','INNING6PLUS_LEAGUE30']:
        for p in np.arange(10.0,60.01,1.0):
            v=vals(st,float(p),den); m=float(v.mean()); s=float(v.std(ddof=0))
            d['opp_bp_ra9_30d_rel']=v
            cf5=coeff.fit_component(d,model,'core_f5',engine.CORE_F5,'runs_f5','league_f5_30d',5.0)
            cl=coeff.fit_component(d,model,'core_late',engine.CORE_LATE,'runs_late_6plus','league_late_30d',300.0)
            md=((m-FM)/FS)**2+((s-FS)/FS)**2
            rows.append({'denominator':den,'p_recent_outs':float(p),'mean':m,'scale':s,'mean_error':m-FM,'scale_error':s-FS,'moment_distance':float(md),'core_sum':float(cf5['param_norm_mae']+cl['param_norm_mae']),'core_f5_norm':cf5['param_norm_mae'],'core_f5_corr':cf5['param_corr'],'core_late_norm':cl['param_norm_mae'],'core_late_corr':cl['param_corr']})
    byden={}
    for den in ['RELIEVER_LEAGUE30','INNING6PLUS_LEAGUE30']:
        z=[x for x in rows if x['denominator']==den]
        byden[den]={'best_moment':sorted(z,key=lambda x:x['moment_distance'])[:5],'best_core':sorted(z,key=lambda x:x['core_sum'])[:5]}
    payload={'version':'mu-v1.2-bp30-reliever-denominator-hybrid-v0.1','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,'numerator':'RELIEVER_STINT_PREGAME_CALENDAR30','prior':'ABSOLUTE_INNING6PLUS_TEAM_LONG','frozen':{'mean':FM,'scale':FS},'scan':{'p_min':10,'p_max':60,'p_step':1},'by_denominator':byden,'decision_rule':'A denominator is preferred only if moment parity and frozen core coefficient parity become directionally more coherent together; no OOS metric used.'}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

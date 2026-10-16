from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import audit_mu_v1_2_bp30_leakage_forensic as src
import audit_mu_v1_2_coefficient_parity as coeff
import audit_mu_v1_2_late_offset_semantics as lateoffset
import audit_mu_v1_2_late_total_variants as lateaudit
import mu_v1_3 as engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'state'/'mu_v1_2_bp30_reliever_p_scan.json'
FROZEN_MEAN=0.9860898056018214
FROZEN_SCALE=0.21158719158114866
P_LONG=360.0


def sufficient_stats(rows,bp6,bprel):
    """Compute row-specific BP30 ingredients once; P scan after this is vectorized."""
    rr=[]; oo=[]; prior=[]; den=[]
    for r in rows.itertuples(index=False):
        d=np.datetime64(r.game_date,'D')
        # pandas filtering retained here only once per row.
        import pandas as pd
        dd=pd.Timestamp(d).normalize(); opp=str(r.opponent)
        preall=bp6[bp6.game_date<dd]
        pre=preall[preall.game_date.dt.year==dd.year]
        if pre.empty: pre=preall
        lg6=src.ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        op=pre[pre.pitching_team.astype(str).eq(opp)]
        rlong=float(op.runs.sum()); olong=float(op.outs.sum())
        team6=(rlong+P_LONG*lg6/27.0)*27.0/(olong+P_LONG) if olong+P_LONG>0 else lg6

        rec=bprel[(bprel.game_date<dd)&(bprel.game_date>=dd-pd.Timedelta(days=30))]
        oppr=rec[rec.pitching_team.astype(str).eq(opp)]
        rr.append(float(oppr.runs.sum())); oo.append(float(oppr.outs.sum()))
        prior.append(float(team6))
        den.append(src.ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg6))
    return tuple(np.asarray(x,float) for x in (rr,oo,prior,den))


def values_for_p(stats,p):
    rr,oo,prior,den=stats
    num=(rr + p*prior/27.0)*27.0/(oo+p)
    return np.divide(num,den,out=np.ones_like(num),where=np.abs(den)>1e-12)


def main():
    model=json.loads(src.MODEL_PATH.read_text(encoding='utf-8'))
    tg,st,bps=src.load_all(); bp6=bps['inning6plus']; bprel=bps['reliever_stint']
    dev=coeff.make_dev_rows(tg,st,bp6)
    dev=lateoffset.apply_offset_only(dev,tg,'late_6plus_pa')
    dev=lateaudit.swap_starter(dev)
    if len(dev)!=2628: raise RuntimeError(f'dev gate {len(dev)}')

    stats=sufficient_stats(dev,bp6,bprel)
    ps=np.arange(35.0,70.01,0.5)
    rows=[]
    d=dev.copy()
    for p in ps:
        vals=values_for_p(stats,float(p))
        m=float(vals.mean()); s=float(vals.std(ddof=0))
        d['opp_bp_ra9_30d_rel']=vals
        cf5=coeff.fit_component(d,model,'core_f5',engine.CORE_F5,'runs_f5','league_f5_30d',5.0)
        cl=coeff.fit_component(d,model,'core_late',engine.CORE_LATE,'runs_late_6plus','league_late_30d',300.0)
        moment_dist=((m-FROZEN_MEAN)/FROZEN_SCALE)**2 + ((s-FROZEN_SCALE)/FROZEN_SCALE)**2
        rows.append({
            'p_recent_outs':float(p),'mean':m,'scale':s,
            'mean_error':m-FROZEN_MEAN,'scale_error':s-FROZEN_SCALE,
            'moment_distance':float(moment_dist),
            'core_f5_norm':cf5['param_norm_mae'],'core_f5_corr':cf5['param_corr'],
            'core_late_norm':cl['param_norm_mae'],'core_late_corr':cl['param_corr'],
            'core_sum':float(cf5['param_norm_mae']+cl['param_norm_mae'])
        })
    payload={
        'version':'mu-v1.2-bp30-reliever-p-scan-v0.2','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,
        'definition':'RELIEVER_STINT_PREGAME_CALENDAR30 / prior=ABSOLUTE_INNING6PLUS_TEAM_LONG',
        'scan':{'min':35.0,'max':70.0,'step':0.5},
        'frozen':{'mean':FROZEN_MEAN,'scale':FROZEN_SCALE},
        'best_by_moment':sorted(rows,key=lambda x:x['moment_distance'])[:10],
        'best_by_coefficient_parity':sorted(rows,key=lambda x:x['core_sum'])[:10],
        'decision_rule':'Prefer a P supported jointly by frozen moment parity and frozen core coefficient parity; no OOS metric is used in this scan.'
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

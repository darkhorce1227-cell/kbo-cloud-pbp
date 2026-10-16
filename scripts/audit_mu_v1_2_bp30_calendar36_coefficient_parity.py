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
MODEL=ROOT/'research'/'mu-v1.2'/'model_spec.json'
OUT=ROOT/'state'/'mu_v1_2_bp30_calendar36_coefficient_parity.json'
FM=0.9860898056018214
FS=0.21158719158114866
P_LONG=360.0


def candidate(rows,bp6,bprel,days,p_recent):
    vals=[]
    for r in rows.itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent)
        preall=bp6[bp6.game_date<d]
        pre=preall[preall.game_date.dt.year==d.year]
        if pre.empty: pre=preall
        lg=src.ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        op=pre[pre.pitching_team.astype(str).eq(opp)]
        team=(float(op.runs.sum())+P_LONG*lg/27.0)*27.0/(float(op.outs.sum())+P_LONG)
        rec=bprel[(bprel.game_date<d)&(bprel.game_date>=d-pd.Timedelta(days=int(days)))]
        o=rec[rec.pitching_team.astype(str).eq(opp)]
        den=src.ra9(float(rec.runs.sum()),float(rec.outs.sum()),lg)
        num=(float(o.runs.sum())+p_recent*team/27.0)*27.0/(float(o.outs.sum())+p_recent)
        vals.append(num/den if den else 1.0)
    return np.asarray(vals,float)


def fit(rows,model,label):
    cf5=coeff.fit_component(rows,model,'core_f5',engine.CORE_F5,'runs_f5','league_f5_30d',5.0)
    cl=coeff.fit_component(rows,model,'core_late',engine.CORE_LATE,'runs_late_6plus','league_late_30d',300.0)
    v=rows['opp_bp_ra9_30d_rel'].to_numpy(float)
    return {
      'variant':label,'mean':float(v.mean()),'scale':float(v.std(ddof=0)),
      'mean_error':float(v.mean()-FM),'scale_error':float(v.std(ddof=0)-FS),
      'core_f5_norm':cf5['param_norm_mae'],'core_f5_corr':cf5['param_corr'],
      'core_late_norm':cl['param_norm_mae'],'core_late_corr':cl['param_corr'],
      'core_sum':float(cf5['param_norm_mae']+cl['param_norm_mae'])
    }


def main():
    model=json.loads(MODEL.read_text(encoding='utf-8'))
    tg,st,bps=src.load_all(); bp6=bps['inning6plus']; rel=bps['reliever_stint']
    dev=coeff.make_dev_rows(tg,st,bp6)
    dev=lateoffset.apply_offset_only(dev,tg,'late_6plus_pa')
    dev=lateaudit.swap_starter(dev)
    if len(dev)!=2628: raise RuntimeError(f'dev gate {len(dev)}')
    out=[fit(dev.copy(),model,'ACTIVE_V0_6_BP30')]
    for label,days,p in [('RELIEVER_CAL30_P39',30,39.0),('RELIEVER_CAL36_P25_5',36,25.5)]:
        d=dev.copy(); d['opp_bp_ra9_30d_rel']=candidate(d,bp6,rel,days,p)
        out.append(fit(d,model,label))
    payload={'version':'mu-v1.2-bp30-calendar36-coeff-parity-v0.1','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,'frozen':{'mean':FM,'scale':FS},'ranked_by_core_sum':sorted(out,key=lambda x:x['core_sum']),'variants':out,'decision_rule':'Calendar36 is eligible for Challenger promotion only if its near-exact frozen moments do not materially degrade frozen core coefficient parity versus the best pregame reliever candidate. No OOS metric used.'}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

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


def main():
    model=json.loads(src.MODEL_PATH.read_text(encoding='utf-8'))
    tg,st,bps=src.load_all(); bp6=bps['inning6plus']; bprel=bps['reliever_stint']
    dev=coeff.make_dev_rows(tg,st,bp6)
    dev=lateoffset.apply_offset_only(dev,tg,'late_6plus_pa')
    dev=lateaudit.swap_starter(dev)
    if len(dev)!=2628: raise RuntimeError(f'dev gate {len(dev)}')

    # Narrow, pre-specified scan only. No OOS optimization.
    ps=np.arange(35.0,70.01,0.5)
    rows=[]
    for p in ps:
        d=dev.copy()
        vals=src.candidate_bp30(d,tg,bp6,bprel,leak=False,p_recent=float(p),prior_mode='abs6')
        m=float(vals.mean()); s=float(vals.std(ddof=0))
        d['opp_bp_ra9_30d_rel']=vals
        # fit coefficients only for the closest moment candidates later would be ideal,
        # but 71 small ridge fits is cheap relative to feature construction.
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
    by_moment=sorted(rows,key=lambda x:x['moment_distance'])[:10]
    by_core=sorted(rows,key=lambda x:x['core_sum'])[:10]
    payload={
        'version':'mu-v1.2-bp30-reliever-p-scan-v0.1','status':'DIAGNOSTIC_ONLY','official_shadow_allowed':False,
        'definition':'RELIEVER_STINT_PREGAME_CALENDAR30 / prior=ABSOLUTE_INNING6PLUS_TEAM_LONG',
        'scan':{'min':35.0,'max':70.0,'step':0.5},
        'frozen':{'mean':FROZEN_MEAN,'scale':FROZEN_SCALE},
        'best_by_moment':by_moment,
        'best_by_coefficient_parity':by_core,
        'decision_rule':'Prefer a P supported jointly by frozen moment parity and frozen core coefficient parity; no OOS metric is used in this scan.'
    }
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

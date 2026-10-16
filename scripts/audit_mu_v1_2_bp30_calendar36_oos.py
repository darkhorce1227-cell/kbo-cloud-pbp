from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered
import audit_mu_v1_2_late_total_variants as lateaudit
import audit_mu_v1_2_late_offset_semantics as lateoffset
import audit_mu_v1_2_bp30_leakage_forensic as bpforensic
import audit_mu_v1_2_bp30_calendar36_coefficient_parity as bp36
import audit_mu_v1_2_30d_longprior_refine as envrefine

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'research'/'mu-v1.2'/'model_spec.json'
OUT=ROOT/'state'/'mu_v1_2_bp30_calendar36_oos.json'
LEAGUE_ENV_P=17.0


def mean_mu(rows,model):
    f5=[]; full=[]
    for _,x in rows.groupby('game_pk'):
        if len(x)!=2: continue
        away=x[x.is_home==0].iloc[0].to_dict(); home=x[x.is_home==1].iloc[0].to_dict()
        p=engine.score(model,away,home)
        f5 += [float(p['f5_away']),float(p['f5_home'])]
        full += [float(p['full_away']),float(p['full_home'])]
    return {'f5':float(np.mean(f5)),'full':float(np.mean(full))}


def eval_variant(rows,model):
    e=engine.evaluate(rows,model); mm=mean_mu(rows,model)
    return {
      'f5':{'mae':e['f5']['team_score_mae'],'nb_nll':e['f5']['nb_nll'],'wdl_brier':e['f5']['wdl_brier_mean3'],'wdl_logloss':e['f5']['wdl_logloss'],'decided_side':e['f5']['decided_side_hit'],'mean_mu':mm['f5']},
      'full':{'mae':e['full']['team_score_mae'],'nb_nll':e['full']['nb_nll'],'wdl_brier':e['full']['wdl_brier_mean3'],'wdl_logloss':e['full']['wdl_logloss'],'decided_side':e['full']['decided_side_hit'],'mean_mu':mm['full']}
    }


def main():
    model=json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    tg,st,bps=bpforensic.load_all(); bp6=bps['inning6plus']; bprel=bps['reliever_stint']
    rows=recovered.make_oos_rows(tg,st,bp6)
    rows=lateaudit.swap_starter(rows)
    rows=lateoffset.apply_offset_only(rows,tg,'late_6plus_pa')
    dates=pd.to_datetime(rows.game_date).dt.normalize().tolist()
    stats=envrefine.precompute(tg,dates)
    f5off,lateoff=envrefine.vectors(dates,stats,LEAGUE_ENV_P)
    rows['league_f5_30d']=f5off; rows['league_late_30d']=lateoff
    if len(rows)!=1368 or rows.game_pk.nunique()!=684: raise RuntimeError('OOS gate')

    old=rows.copy()
    old['opp_bp_ra9_30d_rel']=bpforensic.candidate_bp30(old,tg,bp6,bprel,leak=False,p_recent=39.25,prior_mode='abs6')
    new=rows.copy()
    new['opp_bp_ra9_30d_rel']=bp36.candidate(new,bp6,bprel,36,25.5)
    oldm=eval_variant(old,model); newm=eval_variant(new,model)
    delta={k:{m:float(newm[k][m]-oldm[k][m]) for m in newm[k]} for k in ['f5','full']}
    payload={'version':'mu-v1.2-bp30-calendar36-oos-v0.1','status':'CONFIRMATORY_DIAGNOSTIC_ONLY','official_shadow_allowed':False,'selection_discipline':'Calendar36/P25.5 was selected only from 2024-2025 frozen moment + coefficient parity; 2026 OOS is confirmatory and must not retune parameters.','v0_8_bp30_cal30_p39_25':oldm,'calendar36_p25_5':newm,'delta_new_minus_v0_8':delta}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

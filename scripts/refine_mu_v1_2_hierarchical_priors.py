from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table
import reverse_mu_v1_2_final_development as base

ROOT=Path(__file__).resolve().parents[1]
SHA24=base.SHA24; SHA25=base.SHA25; FROZEN=base.FROZEN


def score(d): return base.score(d)
def sh(s,n,pv,p): return base.shrink(s,n,pv,p)

def prep(training):
    p24=training/'kbo_pbp_2024_training.csv.gz'; p25=training/'kbo_pbp_2025_training.csv.gz'
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24 and hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(df); tg,stc,bp=mu.game_tables(pa); stc=stc.copy(); stc['hbp']=0.0; sta=starter_game_table(pa)
    return base.build_sufficient(tg,stc,sta,bp)

def offense_hier(raw,pl,p30,prior_kind):
    out={k:[] for k in ['off_f5_long_ratio','off_f5_30d_ratio','off_late_long_ratio','off_late_30d_ratio']}
    for r in raw.itertuples(index=False):
      for ph in ['f5','late']:
        ll=getattr(r,f'lg_{ph}_long'); l30=getattr(r,f'lg_{ph}_30')
        long_rate=sh(getattr(r,f'{ph}_sum_long'),r.team_n_long,ll,pl)
        pv=long_rate if prior_kind=='team_long' else l30
        rate30=sh(getattr(r,f'{ph}_sum_30'),r.team_n_30,pv,p30)
        out[f'off_{ph}_long_ratio'].append(long_rate/ll if ll else 1.)
        out[f'off_{ph}_30d_ratio'].append(rate30/l30 if l30 else 1.)
    return {k:np.asarray(v) for k,v in out.items()}

def sp_rates_hier(raw,p_long,p_recent,prior_kind):
    out={k:[] for k in ['sp_kbb_long','sp_kbb_recent5','sp_hr_long','sp_hr_recent5','sp_h_long','sp_h_recent5']}
    for r in raw.itertuples(index=False):
      pden=r.aP_den
      league={'kbb':(r.aP_k-r.aP_bb-r.aP_hbp)/pden if pden else 0.,'hr':r.aP_hr/pden if pden else 0.,'h':r.aP_h/pden if pden else 0.}
      for stat in ['kbb','hr','h']:
        ln={'kbb':r.aL_k-r.aL_bb-r.aL_hbp,'hr':r.aL_hr,'h':r.aL_h}[stat]
        rn={'kbb':r.a5_k-r.a5_bb-r.a5_hbp,'hr':r.a5_hr,'h':r.a5_h}[stat]
        long_val=sh(ln,r.aL_den,league[stat],p_long)
        if prior_kind=='shrunk_long': pv=long_val
        elif prior_kind=='raw_long': pv=ln/r.aL_den if r.aL_den else league[stat]
        else: pv=league[stat]
        recent=sh(rn,r.a5_den,pv,p_recent)
        out[f'sp_{stat}_long'].append(long_val); out[f'sp_{stat}_recent5'].append(recent)
    return {k:np.asarray(v) for k,v in out.items()}

def sp_ip_hier(raw,p_long,p_recent,prior_kind):
    a=[];b=[]
    for r in raw.itertuples(index=False):
      league=r.cP_ip/r.cP_n if r.cP_n else 5.0
      lv=sh(r.cL_ip,r.cL_n,league,p_long)
      if prior_kind=='shrunk_long': pv=lv
      elif prior_kind=='raw_long': pv=r.cL_ip/r.cL_n if r.cL_n else league
      else: pv=league
      rv=sh(r.c5_ip,r.c5_n,pv,p_recent); a.append(lv);b.append(rv)
    return {'sp_ip_long':np.asarray(a),'sp_ip_recent5':np.asarray(b)}

def bp_hier(raw,p_long,p30,scope,prior_kind):
    a=[];b=[]; pr='by' if scope=='season_reset' else 'bc'
    for r in raw.itertuples(index=False):
      tr=getattr(r,f'{pr}L_tr'); to=getattr(r,f'{pr}L_to'); lr=getattr(r,f'{pr}L_lr'); lo=getattr(r,f'{pr}L_lo')
      lg=lr*27/lo if lo else 4.5; long_ra=(tr+p_long*lg/27)*27/(to+p_long) if to+p_long>0 else lg
      tr30=getattr(r,'bc30_tr'); to30=getattr(r,'bc30_to'); lr30=getattr(r,'bc30_lr'); lo30=getattr(r,'bc30_lo'); lg30=lr30*27/lo30 if lo30 else lg
      pv=long_ra if prior_kind=='team_long' else lg30
      ra30=(tr30+p30*pv/27)*27/(to30+p30) if to30+p30>0 else pv
      a.append(long_ra/lg if lg else 1.); b.append(ra30/lg30 if lg30 else 1.)
    return {'opp_bp_ra9_long_rel':np.asarray(a),'opp_bp_ra9_30d_rel':np.asarray(b)}
def best_grid(records): return sorted(records,key=lambda x:x['loss'])[:15]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_hierarchical_prior_refine.json'); z=ap.parse_args(); raw=prep(ROOT/z.training_dir); assert len(raw)==2628
    off=[]
    for kind in ['league','team_long']:
      for pl in np.arange(0,15.01,.5):
       for p30 in np.arange(0,20.01,.5):
        loss,detail=score(offense_hier(raw,float(pl),float(p30),kind)); off.append({'prior':kind,'p_long':float(pl),'p_30d':float(p30),'loss':loss,'detail':detail})
    spr=[]
    for kind in ['league','raw_long','shrunk_long']:
      for pr in range(0,151,5):
        loss,detail=score(sp_rates_hier(raw,150.,float(pr),kind)); spr.append({'prior':kind,'p_long':150.0,'p_recent5':float(pr),'loss':loss,'detail':detail})
    spi=[]
    for kind in ['league','raw_long','shrunk_long']:
      for pr in np.arange(0,10.01,.25):
        loss,detail=score(sp_ip_hier(raw,5.,float(pr),kind)); spi.append({'prior':kind,'p_long':5.0,'p_recent5':float(pr),'loss':loss,'detail':detail})
    bpo=[]
    for scope in ['continuous','season_reset']:
     for kind in ['league','team_long']:
      for pl in range(0,601,15):
       for p30 in range(0,109,9):
        loss,detail=score(bp_hier(raw,float(pl),float(p30),scope,kind)); bpo.append({'scope':scope,'prior30':kind,'p_long':float(pl),'p_30d':float(p30),'loss':loss,'detail':detail})
    payload={'version':'mu-v1.2-hierarchical-prior-refine-v0.1','rows':len(raw),'official_shadow_allowed':False,
             'best':{'offense':best_grid(off)[0],'starter_rates':best_grid(spr)[0],'starter_ip':best_grid(spi)[0],'bullpen':best_grid(bpo)[0],'park_pseudo_games':5.0},
             'top':{'offense':best_grid(off),'starter_rates':best_grid(spr),'starter_ip':best_grid(spi),'bullpen':best_grid(bpo)},
             'note':'Tests semantically plausible hierarchical priors on exact final-development sample. Candidate only.'}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

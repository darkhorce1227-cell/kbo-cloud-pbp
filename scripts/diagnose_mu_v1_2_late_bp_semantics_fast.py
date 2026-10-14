from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT=Path(__file__).resolve().parents[1]
SHA24="d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25="017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"


def loss(a,feat):
    a=np.asarray(a,float); fm=float(FROZEN[feat]['mean']); fs=float(FROZEN[feat]['scale']); m=float(np.nanmean(a)); s=float(np.nanstd(a))
    return {'mean':m,'scale':s,'frozen_mean':fm,'frozen_scale':fs,'loss':abs(m-fm)/fs+abs(s-fs)/fs}

def dev(tg):
    d=tg.game_date; return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(['game_date','game_pk','is_home']).reset_index(drop=True)

def off_suff(tg,target,window):
    rows=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); pre=tg[tg.game_date<d]; my=pre[pre.team.astype(str).eq(str(r.team))]
        if my.empty: my=pre
        lg=float(pre[target].mean()) if len(pre) else 1.0
        if window=='calendar30':
            rec=pre[pre.game_date>=d-pd.Timedelta(days=30)]; myr=rec[rec.team.astype(str).eq(str(r.team))]
            if myr.empty: myr=my.tail(min(30,len(my))); lgr=float(rec[target].mean()) if len(rec) else lg
        else:
            myr=my.tail(30); cutoff=myr.game_date.min() if len(myr) else d; rec=pre[pre.game_date>=cutoff]; lgr=float(rec[target].mean()) if len(rec) else lg
        rows.append((float(my[target].sum()),float(len(my)),lg,float(myr[target].sum()),float(len(myr)),lgr))
    return np.asarray(rows,float)
def off_eval(a,pL,pR):
    L=(a[:,0]+pL*a[:,2])/(a[:,1]+pL)/a[:,2]; R=(a[:,3]+pR*a[:,5])/(a[:,4]+pR)/a[:,5]; return L,R

def bp_suff(tg,bpt,scope,window):
    rows=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); pre=bpt[bpt.game_date<d]
        if scope=='season_reset':
            cur=pre[pre.game_date.dt.year==d.year]
            if len(cur): pre=cur
        opp=pre[pre.pitching_team.astype(str).eq(str(r.opponent))]
        if window=='calendar30':
            rec=bpt[(bpt.game_date<d)&(bpt.game_date>=d-pd.Timedelta(days=30))]; ropp=rec[rec.pitching_team.astype(str).eq(str(r.opponent))]
        else:
            allopp=bpt[(bpt.game_date<d)&(bpt.pitching_team.astype(str).eq(str(r.opponent)))].sort_values('game_date'); ropp=allopp.tail(30); cutoff=ropp.game_date.min() if len(ropp) else d; rec=bpt[(bpt.game_date<d)&(bpt.game_date>=cutoff)]
        rows.append((float(opp.runs.sum()),float(opp.outs.sum()),float(pre.runs.sum()),float(pre.outs.sum()),float(ropp.runs.sum()),float(ropp.outs.sum()),float(rec.runs.sum()),float(rec.outs.sum())))
    return np.asarray(rows,float)
def bp_eval(a,pL,pR):
    lg=np.divide(a[:,2]*27,a[:,3],out=np.full(len(a),4.5),where=a[:,3]>0)
    teamL=np.divide((a[:,0]+pL*lg/27)*27,a[:,1]+pL,out=lg.copy(),where=(a[:,1]+pL)>0)
    lgr=np.divide(a[:,6]*27,a[:,7],out=lg.copy(),where=a[:,7]>0)
    teamR=np.divide((a[:,4]+pR*teamL/27)*27,a[:,5]+pR,out=teamL.copy(),where=(a[:,5]+pR)>0)
    return teamL/lg,teamR/lgr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_late_bp_semantic_diagnostic_fast.json'); z=ap.parse_args()
    td=ROOT/z.training_dir; p24=td/'kbo_pbp_2024_training.csv.gz'; p25=td/'kbo_pbp_2025_training.csv.gz'
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24 and hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True); pa=prepare_pa_fixed(df); tg,_,_=make_team_game(pa); assert len(dev(tg))==2628
    off=[]
    for target in ['late_full_minus_f5','late_6plus_pa','late_6to9_pa']:
      for win in ['calendar30','last30_team_games']:
        s=off_suff(tg,target,win)
        for pL in np.arange(0,21,.25):
          for pR in np.arange(0,16,.25):
            L,R=off_eval(s,float(pL),float(pR)); dL=loss(L,'off_late_long_ratio'); dR=loss(R,'off_late_30d_ratio'); off.append({'target':target,'window':win,'p_long':float(pL),'p_recent':float(pR),'loss':dL['loss']+dR['loss'],'detail':{'off_late_long_ratio':dL,'off_late_30d_ratio':dR}})
    off=sorted(off,key=lambda x:x['loss'])[:40]
    bpts=bp_game_tables(pa,tg); bp=[]
    for definition,bpt in bpts.items():
      for scope in ['continuous','season_reset']:
       for win in ['calendar30','last30_team_games']:
        s=bp_suff(tg,bpt,scope,win)
        for pL in range(180,601,5):
          for pR in range(0,91,3):
            L,R=bp_eval(s,float(pL),float(pR)); dL=loss(L,'opp_bp_ra9_long_rel'); dR=loss(R,'opp_bp_ra9_30d_rel'); bp.append({'definition':definition,'scope':scope,'window':win,'p_long_outs':float(pL),'p_recent_outs':float(pR),'loss':dL['loss']+dR['loss'],'detail':{'opp_bp_ra9_long_rel':dL,'opp_bp_ra9_30d_rel':dR}})
    bp=sorted(bp,key=lambda x:x['loss'])[:50]
    payload={'version':'mu-v1.2-late-bp-semantic-diagnostic-fast-v0.1','rows':2628,'official_shadow_allowed':False,'best':{'late_offense':off[0],'bullpen':bp[0]},'top':{'late_offense':off,'bullpen':bp},'confirmed_exact':{'starter_rates':'all PA; K-BB-HBP; long->league 150 PA; recent5->shrunk long 150 PA'},'note':'Vectorized semantic diagnostic on exact frozen development artifact.'}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

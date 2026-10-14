from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_SHA="d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"


def err(vals,feat):
    a=np.asarray(vals,float); m=float(np.nanmean(a)); s=float(np.nanstd(a)); fm=FROZEN[feat]["mean"]; fs=FROZEN[feat]["scale"]
    return {"mean":m,"scale":s,"frozen_mean":fm,"frozen_scale":fs,"loss":abs(m-fm)/fs+abs(s-fs)/fs}

def score(d):
    q={k:err(v,k) for k,v in d.items()}; return sum(x["loss"] for x in q.values()),q

def games(tg): return tg[(tg.game_date>=TRAIN_START)&(tg.game_date.dt.year==2024)].sort_values(["game_date","game_pk","is_home"])


def sp_suff(tg,st,den_col,hbp):
    smap=st.set_index(["game_pk","pitching_team"])["starter"].to_dict(); rec=[]
    for r in games(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); ss=pd.Timestamp(year=d.year,month=1,day=1); pre=st[(st.game_date<d)&(st.game_date>=ss)]
        sid=smap.get((r.game_pk,r.opponent)); pl=pre[pre.starter.astype(str).eq(str(sid))].sort_values("game_date") if sid is not None else pre.iloc[0:0]
        def one(x):
            den=float(x[den_col].sum()); return den,float(x.k.sum()-x.bb.sum()-(x.hbp.sum() if hbp else 0)),float(x.hr.sum()),float(x.h.sum())
        ld,lk,lhr,lh=one(pl); rd,rk,rhr,rh=one(pl.tail(5)); pden=float(pre[den_col].sum())
        if pden: pk=float(pre.k.sum()-pre.bb.sum()-(pre.hbp.sum() if hbp else 0))/pden; phr=float(pre.hr.sum())/pden; ph=float(pre.h.sum())/pden
        else: pk=phr=ph=0.0
        rec.append((ld,lk,lhr,lh,rd,rk,rhr,rh,pk,phr,ph))
    return np.asarray(rec,float)

def sp_eval(a,pl,pr):
    def v(num_i,den_i,prior_i,p):
        den=a[:,den_i]; num=a[:,num_i]; prior=a[:,prior_i]; return np.where(den+p>0,(num+p*prior)/(den+p),prior)
    return {"sp_kbb_long":v(1,0,8,pl),"sp_hr_long":v(2,0,9,pl),"sp_h_long":v(3,0,10,pl),
            "sp_kbb_recent5":v(5,4,8,pr),"sp_hr_recent5":v(6,4,9,pr),"sp_h_recent5":v(7,4,10,pr)}


def bp_suff(tg,bp,mode):
    b=bp.copy(); b["game_ra9"]=np.where(b.bp_outs>0,b.bp_runs*27/b.bp_outs,np.nan); rows=[]
    for r in games(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); ss=pd.Timestamp(year=d.year,month=1,day=1)
        if mode in ("reliever_pooled","reliever_game_mean"):
            pre=b[(b.game_date<d)&(b.game_date>=ss)]; p30=pre[pre.game_date>=d-pd.Timedelta(days=30)]
            def one(pool):
                team=pool[pool.pitching_team.astype(str).eq(str(r.opponent))]
                if mode=="reliever_pooled":
                    lg=float(pool.bp_runs.sum())*27/float(pool.bp_outs.sum()) if pool.bp_outs.sum()>0 else 4.5
                    return float(team.bp_runs.sum()),float(team.bp_outs.sum()),lg
                lg=float(pool.game_ra9.mean()) if pool.game_ra9.notna().any() else 4.5
                vals=team.game_ra9.dropna(); return float(vals.sum()),float(len(vals)),lg
            rows.append((*one(pre),*one(p30)))
        else:
            hist=tg[(tg.game_date<d)&(tg.game_date>=ss)]; h30=hist[hist.game_date>=d-pd.Timedelta(days=30)]
            def one(pool):
                lg=float(pool.runs_late.mean()) if len(pool) else 1.0; vals=pool[pool.opponent.astype(str).eq(str(r.opponent))].runs_late
                return float(vals.sum()),float(len(vals)),lg
            rows.append((*one(hist),*one(h30)))
    return np.asarray(rows,float)

def bp_eval(a,mode,pl,p30):
    def side(off,p):
        num=a[:,off]; den=a[:,off+1]; lg=a[:,off+2]
        if mode=="reliever_pooled": val=(num+p*lg/27)*27/(den+p); return val/lg
        val=(num+p*lg)/(den+p); return val/lg
    return {"opp_bp_ra9_long_rel":side(0,pl),"opp_bp_ra9_30d_rel":side(3,p30)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--training-dir",default="out_training"); ap.add_argument("--out",default="state/mu_v1_2_sp_bp_definition_diagnostic_fast.json"); z=ap.parse_args()
    p=ROOT/z.training_dir/"kbo_pbp_2024_training.csv.gz"; assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_SHA
    df=pd.read_csv(p,low_memory=False); pa=prepare_pa_fixed(df); tg,_,bp=mu.game_tables(pa); st=starter_game_table(pa)
    sp=[]
    for den in ["pa_classified","pa_all"]:
      for hbp in [False,True]:
        a=sp_suff(tg,st,den,hbp); best=None
        for pl in [0,25,50,75,100,125,150,200]:
          for pr in [0,20,40,60,80,100,125]:
            loss,detail=score(sp_eval(a,float(pl),float(pr))); q={"denominator":den,"subtract_hbp_in_kbb":hbp,"pseudo_long":pl,"pseudo_recent5":pr,"loss":loss,"detail":detail}
            if best is None or loss<best["loss"]: best=q
        sp.append(best)
    bpout=[]
    grids={"reliever_pooled":([0,27,54,81,108],[0,18,36,54]),"reliever_game_mean":([0,1,2,3,5,7,10],[0,1,2,3,5]),"late_team_ra":([0,2,5,10,20],[0,1,2,3,5])}
    for mode,(gl,g30) in grids.items():
        a=bp_suff(tg,bp,mode); best=None
        for pl in gl:
          for pr in g30:
            loss,detail=score(bp_eval(a,mode,float(pl),float(pr))); q={"mode":mode,"pseudo_long":pl,"pseudo_30d":pr,"loss":loss,"detail":detail}
            if best is None or loss<best["loss"]: best=q
        bpout.append(best)
    payload={"version":"mu-v1.2-sp-bp-definition-diagnostic-fast-v0.1","official_shadow_allowed":False,
             "starter_variants_ranked":sorted(sp,key=lambda x:x["loss"]),"bullpen_variants_ranked":sorted(bpout,key=lambda x:x["loss"])}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

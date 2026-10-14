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
SHA24="d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25="017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"
HITS={"single","double","triple","home_run"}


def metric(vals,feat):
    a=np.asarray(vals,float); m=float(np.nanmean(a)); s=float(np.nanstd(a)); fm=FROZEN[feat]["mean"]; fs=FROZEN[feat]["scale"]
    return {"n":int(np.isfinite(a).sum()),"mean":m,"scale":s,"frozen_mean":fm,"frozen_scale":fs,
            "abs_mean_diff":abs(m-fm),"abs_scale_diff":abs(s-fs),"loss":abs(m-fm)/fs+abs(s-fs)/fs}


def dev_games(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"])


def hist(df,d,scope):
    pre=df[df.game_date<d]
    if scope=="season_reset":
        cur=pre[pre.game_date.dt.year==d.year]
        return cur if len(cur) else pre
    return pre


def make_features(tg,st_class,st_all,bp,scope,starter_mode,park_pseudo=0.0):
    smap=st_class.set_index(["game_pk","pitching_team"])["starter"].to_dict()
    # st_all carries pa_all/hbp; st_class has IP and classified PA rates.
    ipmap=st_class.set_index(["game_pk","pitching_team","starter"])["ip"].to_dict()
    if starter_mode=="all_pa_hbp":
        st=st_all.copy(); st["ip"]=[ipmap.get((r.game_pk,r.pitching_team,r.starter),np.nan) for r in st.itertuples(index=False)]
        den_col="pa_all"
    else:
        st=st_class.copy(); den_col="pa"
        st["hbp"]=0.0
    out={k:[] for k in FROZEN}
    for r in dev_games(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date)
        tpre=hist(tg,d,scope); t30=tg[(tg.game_date<d)&(tg.game_date>=d-pd.Timedelta(days=30))]
        if t30.empty: t30=tpre
        my=tpre[tpre.team.astype(str).eq(str(r.team))]; my30=t30[t30.team.astype(str).eq(str(r.team))]
        if my.empty: my=tpre
        if my30.empty: my30=my
        lf5=float(tpre.runs_f5.mean()); lf530=float(t30.runs_f5.mean()); llate=float(tpre.runs_late.mean()); llate30=float(t30.runs_late.mean())
        out["off_f5_long_ratio"].append(float(my.runs_f5.mean())/lf5 if lf5 else 1)
        out["off_f5_30d_ratio"].append(float(my30.runs_f5.mean())/lf530 if lf530 else 1)
        out["off_late_long_ratio"].append(float(my.runs_late.mean())/llate if llate else 1)
        out["off_late_30d_ratio"].append(float(my30.runs_late.mean())/llate30 if llate30 else 1)
        out["is_home"].append(float(r.is_home))

        sid=smap.get((r.game_pk,r.opponent)); spre=hist(st,d,scope); player=spre[spre.starter.astype(str).eq(str(sid))].sort_values("game_date") if sid is not None else spre.iloc[0:0]
        def rates(x):
            den=float(x[den_col].sum()); lden=float(spre[den_col].sum())
            hbp=float(x.hbp.sum()) if starter_mode=="all_pa_hbp" else 0.0; lhbp=float(spre.hbp.sum()) if starter_mode=="all_pa_hbp" else 0.0
            prior_kbb=float(spre.k.sum()-spre.bb.sum()-lhbp)/lden if lden else 0.0
            prior_hr=float(spre.hr.sum())/lden if lden else 0.0; prior_h=float(spre.h.sum())/lden if lden else 0.0
            return ((float(x.k.sum()-x.bb.sum()-hbp)/den if den else prior_kbb),
                    (float(x.hr.sum())/den if den else prior_hr),(float(x.h.sum())/den if den else prior_h))
        k,hrr,hh=rates(player); k5,hr5,h5=rates(player.tail(5))
        for key,val in [("sp_kbb_long",k),("sp_hr_long",hrr),("sp_h_long",hh),("sp_kbb_recent5",k5),("sp_hr_recent5",hr5),("sp_h_recent5",h5)]: out[key].append(val)
        lip=float(spre.ip.mean()) if len(spre) else 5.0
        out["sp_ip_long"].append(float(player.ip.mean()) if len(player) else lip)
        out["sp_ip_recent5"].append(float(player.tail(5).ip.mean()) if len(player) else lip)

        bpre=hist(bp,d,scope); b30=bp[(bp.game_date<d)&(bp.game_date>=d-pd.Timedelta(days=30))]
        if b30.empty: b30=bpre
        def brel(pool):
            lg=float(pool.bp_runs.sum())*27/float(pool.bp_outs.sum()) if pool.bp_outs.sum()>0 else 4.5
            team=pool[pool.pitching_team.astype(str).eq(str(r.opponent))]
            if team.bp_outs.sum()>0: val=float(team.bp_runs.sum())*27/float(team.bp_outs.sum())
            else: val=lg
            return val/lg if lg else 1.0
        out["opp_bp_ra9_long_rel"].append(brel(bpre)); out["opp_bp_ra9_30d_rel"].append(brel(b30))

        home=r.team if int(r.is_home) else r.opponent
        pgids=tpre[(tpre.team.astype(str).eq(str(home)))&(tpre.is_home==1)].game_pk.unique(); pg=tpre[tpre.game_pk.isin(pgids)]
        lg=float(tpre.runs_full.mean()); n=float(len(pgids))
        if n+park_pseudo>0:
            pr=(float(pg.runs_full.sum())+2*park_pseudo*lg)/(2*(n+park_pseudo)); pf=pr/lg if lg else 1.0
        else: pf=1.0
        out["park_factor"].append(pf)
    return {k:np.asarray(v,float) for k,v in out.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--training-dir",default="out_training"); ap.add_argument("--out",default="state/mu_v1_2_final_development_audit.json"); z=ap.parse_args()
    p24=ROOT/z.training_dir/"kbo_pbp_2024_training.csv.gz"; p25=ROOT/z.training_dir/"kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24 and hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True); pa=prepare_pa_fixed(df); tg,st,bp=mu.game_tables(pa); sta=starter_game_table(pa)
    candidates=[]
    for scope in ["continuous","season_reset"]:
      for sm in ["classified_pa_kbb","all_pa_hbp"]:
       for pp in [0.0,5.0,20.0]:
        f=make_features(tg,st,sta,bp,scope,sm,pp); checks={k:metric(v,k) for k,v in f.items()}; loss=sum(x["loss"] for x in checks.values())
        candidates.append({"history_scope":scope,"starter_mode":sm,"park_pseudo_games":pp,"rows":len(next(iter(f.values()))),"loss":loss,"checks":checks})
    candidates.sort(key=lambda x:x["loss"])
    payload={"version":"mu-v1.2-final-development-audit-v0.1","expected_rows":2628,"official_shadow_allowed":False,"candidates_ranked":candidates}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

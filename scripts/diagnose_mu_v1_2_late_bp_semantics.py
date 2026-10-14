from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START

ROOT = Path(__file__).resolve().parents[1]
SHA24 = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25 = "017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"


def moment_loss(vals, feat):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    m = float(a.mean()); s = float(a.std(ddof=0))
    fm = float(FROZEN[feat]["mean"]); fs = float(FROZEN[feat]["scale"])
    return {
        "mean": m, "scale": s, "frozen_mean": fm, "frozen_scale": fs,
        "mean_error": m-fm, "scale_error": s-fs,
        "loss": abs(m-fm)/max(fs,1e-12)+abs(s-fs)/max(fs,1e-12),
    }


def dev_rows(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"]).reset_index(drop=True)


def make_team_game(pa):
    tg, st, bp = mu.game_tables(pa)
    # Alternative late-run definitions from PA event scoring.
    for label, mask in {
        "late_6plus_pa": pa["inning"] >= 6,
        "late_6to9_pa": (pa["inning"] >= 6) & (pa["inning"] <= 9),
    }.items():
        x = (pa[mask].groupby(["game_pk","batting_team"],as_index=False)
             .agg(v=("runs_scored","sum")))
        m={(str(r.game_pk),str(r.batting_team)):float(r.v) for r in x.itertuples(index=False)}
        tg[label]=[m.get((str(r.game_pk),str(r.team)),0.0) for r in tg.itertuples(index=False)]
    tg["late_full_minus_f5"] = tg["runs_late"].astype(float)
    return tg, st, bp


def offense_variant(tg, target_col, window_mode, p_long, p_recent):
    outL=[]; outR=[]
    for r in dev_rows(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date)
        pre=tg[tg.game_date<d]
        my=pre[pre.team.astype(str).eq(str(r.team))]
        if my.empty: my=pre
        lg=float(pre[target_col].mean()) if len(pre) else 0.0
        if window_mode=="calendar30":
            recent=pre[pre.game_date>=d-pd.Timedelta(days=30)]
            myr=recent[recent.team.astype(str).eq(str(r.team))]
            if myr.empty: myr=my.tail(min(30,len(my)))
            lgr=float(recent[target_col].mean()) if len(recent) else lg
        elif window_mode=="last30_team_games":
            myr=my.tail(30)
            cutoff=myr.game_date.min() if len(myr) else d
            recent=pre[pre.game_date>=cutoff]
            lgr=float(recent[target_col].mean()) if len(recent) else lg
        else: raise ValueError(window_mode)
        long=(float(my[target_col].sum())+p_long*lg)/(len(my)+p_long) if len(my)+p_long>0 else lg
        rec=(float(myr[target_col].sum())+p_recent*lgr)/(len(myr)+p_recent) if len(myr)+p_recent>0 else lgr
        outL.append(long/lg if lg else 1.0)
        outR.append(rec/lgr if lgr else 1.0)
    return np.asarray(outL),np.asarray(outR)


def starter_map(pa):
    first=(pa.sort_values(["game_date","game_pk","inning","at_bat_number"])
           .groupby(["game_pk","pitching_team"],as_index=False).first()
           [["game_pk","pitching_team","pitcher"]].rename(columns={"pitcher":"starter"}))
    return {(str(r.game_pk),str(r.pitching_team)):str(r.starter) for r in first.itertuples(index=False)}


def bp_game_tables(pa, tg):
    smap=starter_map(pa)
    p=pa.copy()
    p["starter"]=[smap.get((str(r.game_pk),str(r.pitching_team)),"") for r in p.itertuples(index=False)]
    p["nonstarter"]=p["pitcher"].astype(str)!=p["starter"].astype(str)
    defs={
        "reliever_stint": p["nonstarter"],
        "inning6plus": p["inning"]>=6,
        "inning6to9": (p["inning"]>=6)&(p["inning"]<=9),
    }
    result={}
    for name,mask in defs.items():
        z=(p[mask].groupby(["game_pk","game_date","pitching_team"],as_index=False)
           .agg(runs=("runs_scored","sum"),outs=("outs_recorded","sum")))
        # Include zero-exposure team-games so historical team counts stay aligned.
        rows=[]
        for r in tg[tg.is_home==1].itertuples(index=False):
            for team in (r.team,r.opponent):
                hit=z[(z.game_pk.astype(str)==str(r.game_pk))&(z.pitching_team.astype(str)==str(team))]
                if len(hit): rr=float(hit.runs.iloc[0]); oo=float(hit.outs.iloc[0])
                else: rr=0.0; oo=0.0
                rows.append((r.game_pk,r.game_date,team,rr,oo))
        result[name]=pd.DataFrame(rows,columns=["game_pk","game_date","pitching_team","runs","outs"])
    return result


def bp_features(tg,bpt, scope, window_mode, p_long, p_recent):
    valsL=[];valsR=[]
    for r in dev_rows(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date)
        pre=bpt[bpt.game_date<d]
        if scope=="season_reset":
            cur=pre[pre.game_date.dt.year==d.year]
            if len(cur): pre=cur
        opp=pre[pre.pitching_team.astype(str).eq(str(r.opponent))]
        lr=float(pre.runs.sum()); lo=float(pre.outs.sum())
        lg=lr*27/lo if lo>0 else 4.5
        tr=float(opp.runs.sum()); to=float(opp.outs.sum())
        teamL=(tr+p_long*lg/27)*27/(to+p_long) if to+p_long>0 else lg
        if window_mode=="calendar30":
            rec_all=bpt[(bpt.game_date<d)&(bpt.game_date>=d-pd.Timedelta(days=30))]
            rec_opp=rec_all[rec_all.pitching_team.astype(str).eq(str(r.opponent))]
        elif window_mode=="last30_team_games":
            opp_all=bpt[(bpt.game_date<d)&(bpt.pitching_team.astype(str).eq(str(r.opponent)))].sort_values("game_date")
            rec_opp=opp_all.tail(30)
            cutoff=rec_opp.game_date.min() if len(rec_opp) else d
            rec_all=bpt[(bpt.game_date<d)&(bpt.game_date>=cutoff)]
        else: raise ValueError(window_mode)
        rlr=float(rec_all.runs.sum()); rlo=float(rec_all.outs.sum())
        lgr=rlr*27/rlo if rlo>0 else lg
        rr=float(rec_opp.runs.sum()); ro=float(rec_opp.outs.sum())
        # Hierarchical recent prior = team's shrunk long RA9.
        teamR=(rr+p_recent*teamL/27)*27/(ro+p_recent) if ro+p_recent>0 else teamL
        valsL.append(teamL/lg if lg else 1.0)
        valsR.append(teamR/lgr if lgr else 1.0)
    return np.asarray(valsL),np.asarray(valsR)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_late_bp_semantic_diagnostic.json'); a=ap.parse_args()
    td=ROOT/a.training_dir; p24=td/'kbo_pbp_2024_training.csv.gz';p25=td/'kbo_pbp_2025_training.csv.gz'
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(df); tg,_,_=make_team_game(pa); assert len(dev_rows(tg))==2628

    off=[]
    for target in ['late_full_minus_f5','late_6plus_pa','late_6to9_pa']:
      for win in ['calendar30','last30_team_games']:
       for pl in np.arange(0,21,.5):
        for pr in np.arange(0,16,.5):
            L,R=offense_variant(tg,target,win,float(pl),float(pr))
            dl=moment_loss(L,'off_late_long_ratio');dr=moment_loss(R,'off_late_30d_ratio')
            off.append({'target':target,'window':win,'p_long':float(pl),'p_recent':float(pr),'loss':dl['loss']+dr['loss'],'detail':{'off_late_long_ratio':dl,'off_late_30d_ratio':dr}})
    off=sorted(off,key=lambda x:x['loss'])[:30]

    bpts=bp_game_tables(pa,tg); bp=[]
    for definition,bpt in bpts.items():
      for scope in ['continuous','season_reset']:
       for win in ['calendar30','last30_team_games']:
        for pl in range(180,601,15):
         for pr in range(0,91,3):
            L,R=bp_features(tg,bpt,scope,win,float(pl),float(pr))
            dl=moment_loss(L,'opp_bp_ra9_long_rel');dr=moment_loss(R,'opp_bp_ra9_30d_rel')
            bp.append({'definition':definition,'scope':scope,'window':win,'p_long_outs':float(pl),'p_recent_outs':float(pr),'loss':dl['loss']+dr['loss'],'detail':{'opp_bp_ra9_long_rel':dl,'opp_bp_ra9_30d_rel':dr}})
    bp=sorted(bp,key=lambda x:x['loss'])[:40]
    payload={'version':'mu-v1.2-late-bp-semantic-diagnostic-v0.1','rows':2628,'official_shadow_allowed':False,'best':{'late_offense':off[0],'bullpen':bp[0]},'top':{'late_offense':off,'bullpen':bp},'confirmed_exact':{'starter_rates':'all PA denominator; K-BB-HBP; long -> league with 150 PA; recent5 -> shrunk long with 150 PA'},'note':'Semantic candidate audit only. Exact or near-exact frozen moments are necessary but not sufficient for production promotion.'}
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

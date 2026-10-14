from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"


def merr(vals, feat):
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    m=float(a.mean()); s=float(a.std(ddof=0)); fm=FROZEN[feat]["mean"]; fs=FROZEN[feat]["scale"]
    return {"mean":m,"scale":s,"frozen_mean":fm,"frozen_scale":fs,
            "loss":abs(m-fm)/max(fs,1e-12)+abs(s-fs)/max(fs,1e-12)}


def score(features):
    detail={k:merr(v,k) for k,v in features.items()}
    return float(sum(v["loss"] for v in detail.values())), detail


def starter_game_table(pa):
    first=(pa.sort_values(["game_date","game_pk","inning","at_bat_number"])
           .groupby(["game_pk","pitching_team"],as_index=False).first()
           [["game_pk","pitching_team","pitcher","game_date"]].rename(columns={"pitcher":"starter"}))
    p=pa.merge(first[["game_pk","pitching_team","starter"]],on=["game_pk","pitching_team"],how="left")
    p=p[p["pitcher"].astype(str).eq(p["starter"].astype(str))].copy()
    p["pa_all"]=1
    p["pa_classified"]=p["event"].notna().astype(int)
    p["k"]=p["event"].eq("strikeout").astype(int)
    p["bb"]=p["event"].eq("walk").astype(int)
    p["hbp"]=p["event"].eq("hit_by_pitch").astype(int)
    p["hr"]=p["event"].eq("home_run").astype(int)
    p["h"]=p["event"].isin({"single","double","triple","home_run"}).astype(int)
    return (p.groupby(["game_pk","game_date","pitching_team","starter"],as_index=False)
            .agg(pa_all=("pa_all","sum"),pa_classified=("pa_classified","sum"),k=("k","sum"),bb=("bb","sum"),
                 hbp=("hbp","sum"),hr=("hr","sum"),h=("h","sum")))


def starter_features(tg, st_alt, denom_col, subtract_hbp, pseudo_long, pseudo_recent):
    st_map=st_alt.set_index(["game_pk","pitching_team"])["starter"].to_dict()
    out={k:[] for k in ["sp_kbb_long","sp_kbb_recent5","sp_hr_long","sp_hr_recent5","sp_h_long","sp_h_recent5"]}
    games=tg[(tg.game_date>=TRAIN_START)&(tg.game_date.dt.year==2024)].sort_values(["game_date","game_pk","is_home"])
    for r in games.itertuples(index=False):
        d=pd.Timestamp(r.game_date); ss=pd.Timestamp(year=d.year,month=1,day=1)
        pre=st_alt[(st_alt.game_date<d)&(st_alt.game_date>=ss)]
        sid=st_map.get((r.game_pk,r.opponent)); player=pre[pre.starter.astype(str).eq(str(sid))].sort_values("game_date") if sid is not None else pre.iloc[0:0]
        for tag,x,pseudo,suffix in [("long",player,pseudo_long,"long"),("recent",player.tail(5),pseudo_recent,"recent5")]:
            den=float(x[denom_col].sum()); lden=float(pre[denom_col].sum())
            kbb_num=float(x.k.sum()-x.bb.sum()-(x.hbp.sum() if subtract_hbp else 0))
            lkbb=float(pre.k.sum()-pre.bb.sum()-(pre.hbp.sum() if subtract_hbp else 0))
            priors={"kbb":lkbb/lden if lden else 0.0,"hr":float(pre.hr.sum())/lden if lden else 0.0,"h":float(pre.h.sum())/lden if lden else 0.0}
            nums={"kbb":kbb_num,"hr":float(x.hr.sum()),"h":float(x.h.sum())}
            for stat in ("kbb","hr","h"):
                val=(nums[stat]+pseudo*priors[stat])/(den+pseudo) if den+pseudo>0 else priors[stat]
                out[f"sp_{stat}_{suffix}"].append(val)
    return {k:np.asarray(v) for k,v in out.items()}


def best_starter(tg, st_alt):
    variants=[]; long_grid=[0,25,50,75,100,125,150,200]; recent_grid=[0,20,40,60,80,100,125]
    for den in ["pa_classified","pa_all"]:
        for hbp in [False,True]:
            best=None
            for pl in long_grid:
                for pr in recent_grid:
                    feats=starter_features(tg,st_alt,den,hbp,float(pl),float(pr)); loss,detail=score(feats)
                    rec={"denominator":den,"subtract_hbp_in_kbb":hbp,"pseudo_long":pl,"pseudo_recent5":pr,"loss":loss,"detail":detail}
                    if best is None or loss<best["loss"]: best=rec
            variants.append(best)
    return sorted(variants,key=lambda x:x["loss"])


def bp_current_table(pa):
    tg,_,bp=mu.game_tables(pa)
    bp=bp.copy(); bp["game_ra9"]=np.where(bp.bp_outs>0,bp.bp_runs*27.0/bp.bp_outs,np.nan)
    return tg,bp


def bp_features(tg,bp,mode,pseudo_long,pseudo_30):
    games=tg[(tg.game_date>=TRAIN_START)&(tg.game_date.dt.year==2024)].sort_values(["game_date","game_pk","is_home"])
    a=[]; b=[]
    for r in games.itertuples(index=False):
        d=pd.Timestamp(r.game_date); ss=pd.Timestamp(year=d.year,month=1,day=1)
        if mode.startswith("reliever_"):
            pre=bp[(bp.game_date<d)&(bp.game_date>=ss)]; p30=pre[pre.game_date>=d-pd.Timedelta(days=30)]
            opp=pre[pre.pitching_team.astype(str).eq(str(r.opponent))]; opp30=p30[p30.pitching_team.astype(str).eq(str(r.opponent))]
            def calc(pool,team,pseudo):
                if mode=="reliever_pooled":
                    lr=float(pool.bp_runs.sum())*27/float(pool.bp_outs.sum()) if pool.bp_outs.sum()>0 else 4.5
                    rr=float(team.bp_runs.sum()); oo=float(team.bp_outs.sum()); prior=pseudo*lr/27
                    tr=(rr+prior)*27/(oo+pseudo) if oo+pseudo>0 else lr
                else:
                    lg=float(pool.game_ra9.dropna().mean()) if pool.game_ra9.notna().any() else 4.5
                    vals=team.game_ra9.dropna(); n=len(vals); tr=(float(vals.sum())+pseudo*lg)/(n+pseudo) if n+pseudo>0 else lg; lr=lg
                return tr/lr if lr else 1.0
            a.append(calc(pre,opp,pseudo_long)); b.append(calc(p30,opp30,pseudo_30))
        else:
            # Team runs allowed in innings 6+; each batting row maps to the opposing pitching team.
            hist=tg[(tg.game_date<d)&(tg.game_date>=ss)].copy(); h30=hist[hist.game_date>=d-pd.Timedelta(days=30)]
            def calc(pool,pseudo):
                lg=float(pool.runs_late.mean()) if len(pool) else 0.0
                vals=pool[pool.opponent.astype(str).eq(str(r.opponent))].runs_late
                n=len(vals); tr=(float(vals.sum())+pseudo*lg)/(n+pseudo) if n+pseudo>0 else lg
                return tr/lg if lg else 1.0
            a.append(calc(hist,pseudo_long)); b.append(calc(h30,pseudo_30))
    return {"opp_bp_ra9_long_rel":np.asarray(a),"opp_bp_ra9_30d_rel":np.asarray(b)}


def best_bp(tg,bp):
    variants=[]
    grids={"reliever_pooled":([0,27,54,81,108],[0,18,36,54]),"reliever_game_mean":([0,1,2,3,5,7,10],[0,1,2,3,5]),"late_team_ra":([0,2,5,10,20],[0,1,2,3,5])}
    for mode,(gl,g30) in grids.items():
        best=None
        for pl in gl:
            for pr in g30:
                feats=bp_features(tg,bp,mode,float(pl),float(pr)); loss,detail=score(feats)
                rec={"mode":mode,"pseudo_long":pl,"pseudo_30d":pr,"loss":loss,"detail":detail}
                if best is None or loss<best["loss"]: best=rec
        variants.append(best)
    return sorted(variants,key=lambda x:x["loss"])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--training-dir",default="out_training"); ap.add_argument("--out",default="state/mu_v1_2_sp_bp_definition_diagnostic.json"); args=ap.parse_args()
    p=ROOT/args.training_dir/"kbo_pbp_2024_training.csv.gz"
    if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED_SHA: raise RuntimeError("frozen artifact SHA mismatch")
    df=pd.read_csv(p,low_memory=False); pa=prepare_pa_fixed(df); tg,bp=bp_current_table(pa); st_alt=starter_game_table(pa)
    sp=best_starter(tg,st_alt); bpres=best_bp(tg,bp)
    payload={"version":"mu-v1.2-sp-bp-definition-diagnostic-v0.1","official_shadow_allowed":False,
             "starter_variants_ranked":sp,"bullpen_variants_ranked":bpres,
             "note":"Only semantically plausible definitions were tested. Ranking is diagnostic, not permission to promote."}
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

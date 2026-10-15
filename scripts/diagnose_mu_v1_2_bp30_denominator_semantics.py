from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_late_bp_semantics import make_team_game, bp_game_tables

ROOT = Path(__file__).resolve().parents[1]
SHA24 = "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25 = "017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"
P_LONG = 360.0
P_RECENT = 54.0
FEAT = "opp_bp_ra9_30d_rel"
OUT = ROOT / "state" / "mu_v1_2_bp30_denominator_semantics.json"


def dev(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START)) | (d.dt.year==2025)].sort_values(["game_date","game_pk","is_home"]).reset_index(drop=True)


def safe_ra9(runs, outs, fallback=np.nan):
    return float(runs*27.0/outs) if outs and outs>0 else float(fallback)


def moment(vals):
    a=np.asarray(vals,float)
    fm=float(FROZEN[FEAT]["mean"]); fs=float(FROZEN[FEAT]["scale"])
    m=float(np.nanmean(a)); s=float(np.nanstd(a,ddof=0))
    return {"mean":m,"scale":s,"frozen_mean":fm,"frozen_scale":fs,"mean_error":m-fm,"scale_error":s-fs,"loss":abs(m-fm)/fs+abs(s-fs)/fs}


def load():
    td=ROOT/"out_training"
    p24=td/"kbo_pbp_2024_training.csv.gz"; p25=td/"kbo_pbp_2025_training.csv.gz"
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24
    assert hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    raw=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True)
    pa=prepare_pa_fixed(raw)
    tg,_,_=make_team_game(pa)
    bp=bp_game_tables(pa,tg)["inning6plus"].copy()
    return tg,bp


def team_raw_baselines(recent, fallback):
    g=recent.groupby("pitching_team",as_index=False).agg(runs=("runs","sum"),outs=("outs","sum"))
    x=np.array([safe_ra9(float(r.runs),float(r.outs),np.nan) for r in g.itertuples(index=False)],float)
    x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else fallback


def game_raw_baseline(recent, fallback):
    vals=[]
    for r in recent.itertuples(index=False):
        if float(r.outs)>0:
            vals.append(float(r.runs)*27.0/float(r.outs))
    return float(np.mean(vals)) if vals else fallback


def build(tg,bp):
    vals={k:[] for k in [
        "pooled_league30",
        "equal_team_raw_league30",
        "equal_game_raw_league30",
        "equal_team_shrunk_teamlong_league30",
        "equal_team_shrunk_league30_league30",
        "pooled_league30_prevday_exclusive",
        "pooled_league30_30teamgames",
    ]}
    diagnostics=[]
    for r in dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date); opp=str(r.opponent)
        pre_all=bp[bp.game_date<d]
        pre=pre_all[pre_all.game_date.dt.year==d.year]
        if pre.empty: pre=pre_all
        opp_long=pre[pre.pitching_team.astype(str).eq(opp)]
        lg_long=safe_ra9(float(pre.runs.sum()),float(pre.outs.sum()),4.5)
        tr=float(opp_long.runs.sum()); to=float(opp_long.outs.sum())
        team_long=(tr+P_LONG*lg_long/27.0)*27.0/(to+P_LONG) if to+P_LONG>0 else lg_long

        recent=bp[(bp.game_date<d)&(bp.game_date>=d-pd.Timedelta(days=30))]
        opp_recent=recent[recent.pitching_team.astype(str).eq(opp)]
        rr=float(opp_recent.runs.sum()); ro=float(opp_recent.outs.sum())
        lg30=safe_ra9(float(recent.runs.sum()),float(recent.outs.sum()),lg_long)
        team_recent=(rr+P_RECENT*team_long/27.0)*27.0/(ro+P_RECENT) if ro+P_RECENT>0 else team_long

        den_teamraw=team_raw_baselines(recent,lg30)
        den_gameraw=game_raw_baseline(recent,lg30)

        # Equal-team mean after applying the same recent shrink to each club.
        shr_teamlong=[]; shr_lg30=[]
        teams=sorted(pre.pitching_team.astype(str).unique())
        for tm in teams:
            pl=pre[pre.pitching_team.astype(str).eq(tm)]
            tlr=float(pl.runs.sum()); tlo=float(pl.outs.sum())
            tl=(tlr+P_LONG*lg_long/27.0)*27.0/(tlo+P_LONG) if tlo+P_LONG>0 else lg_long
            pr=recent[recent.pitching_team.astype(str).eq(tm)]
            prr=float(pr.runs.sum()); pro=float(pr.outs.sum())
            shr_teamlong.append((prr+P_RECENT*tl/27.0)*27.0/(pro+P_RECENT) if pro+P_RECENT>0 else tl)
            shr_lg30.append((prr+P_RECENT*lg30/27.0)*27.0/(pro+P_RECENT) if pro+P_RECENT>0 else lg30)
        den_shr_team=float(np.mean(shr_teamlong)) if shr_teamlong else lg30
        den_shr_lg=float(np.mean(shr_lg30)) if shr_lg30 else lg30

        # 30-day boundary alternative: > d-30 rather than >= d-30.
        rec_ex=bp[(bp.game_date<d)&(bp.game_date>d-pd.Timedelta(days=30))]
        den_ex=safe_ra9(float(rec_ex.runs.sum()),float(rec_ex.outs.sum()),lg_long)
        opp_ex=rec_ex[rec_ex.pitching_team.astype(str).eq(opp)]
        exr=float(opp_ex.runs.sum()); exo=float(opp_ex.outs.sum())
        num_ex=(exr+P_RECENT*team_long/27.0)*27.0/(exo+P_RECENT) if exo+P_RECENT>0 else team_long

        # Last 30 team bullpen games for numerator; league denominator from union of last-30 team games for all clubs.
        rec30_parts=[]
        for tm in sorted(pre.pitching_team.astype(str).unique()):
            z=pre[pre.pitching_team.astype(str).eq(tm)].sort_values(["game_date","game_pk"]).tail(30)
            rec30_parts.append(z)
        rec30g=pd.concat(rec30_parts,ignore_index=True) if rec30_parts else pre.iloc[0:0]
        den_30g=safe_ra9(float(rec30g.runs.sum()),float(rec30g.outs.sum()),lg_long)
        opp30g=pre[pre.pitching_team.astype(str).eq(opp)].sort_values(["game_date","game_pk"]).tail(30)
        r30=float(opp30g.runs.sum()); o30=float(opp30g.outs.sum())
        num_30g=(r30+P_RECENT*team_long/27.0)*27.0/(o30+P_RECENT) if o30+P_RECENT>0 else team_long

        pairs={
            "pooled_league30":(team_recent,lg30),
            "equal_team_raw_league30":(team_recent,den_teamraw),
            "equal_game_raw_league30":(team_recent,den_gameraw),
            "equal_team_shrunk_teamlong_league30":(team_recent,den_shr_team),
            "equal_team_shrunk_league30_league30":(team_recent,den_shr_lg),
            "pooled_league30_prevday_exclusive":(num_ex,den_ex),
            "pooled_league30_30teamgames":(num_30g,den_30g),
        }
        for k,(num,den) in pairs.items(): vals[k].append(num/den if den else 1.0)
        diagnostics.append({"game_date":str(d.date()),"pooled":lg30,"equal_team_raw":den_teamraw,"equal_game_raw":den_gameraw,"equal_team_shrunk_teamlong":den_shr_team})
    return vals,diagnostics


def main():
    tg,bp=load(); vals,diag=build(tg,bp)
    ranked=[{"variant":k,**moment(v)} for k,v in vals.items()]
    ranked.sort(key=lambda x:x["loss"])
    d=pd.DataFrame(diag)
    baseline_summary={c:{"mean":float(d[c].mean()),"median":float(d[c].median())} for c in ["pooled","equal_team_raw","equal_game_raw","equal_team_shrunk_teamlong"]}
    payload={"version":"mu-v1.2-bp30-denominator-semantics-v0.1","status":"DIAGNOSTIC_ONLY","official_shadow_allowed":False,"rows":2628,"ranked":ranked,"baseline_summary":baseline_summary,"note":"Tests plausible league-baseline aggregation/window semantics without arbitrary multiplicative fitting."}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

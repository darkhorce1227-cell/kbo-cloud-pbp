#!/usr/bin/env python3
"""Derive a *market-unseen* 10/20 KIA@LG 2026 PS model probability grid.

This is NOT a sportsbook odds lookup. Its inputs are a frozen μ-v1.2
challenger snapshot and 2024-25 fitted fixed NB dispersion from model_spec.
Independent negative-binomial team scores are an approximate 9-inning
distribution, not official post-extra-innings settling prices. Exclude from
real-bet EV/ROI unless market mapping, extra innings and quote available
timestamps have been validated separately.

Snapshot MUST be newly created and externally ADD-committed before kickoff.
The original snapshot itself must precede the first pitch. No retrospective
or scenario-edited matchup can be represented as the original CLEAN PRE.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
INPUT_DIR=ROOT/"state/prospective_mu_v1_2"
OUTPUT=ROOT/"research/market_fair_odds/pregame_mu_grid_2026_10_20_HT_LG.json"
MODEL=ROOT/"research/mu-v1.2/model_spec.json"
KICKOFF=datetime.fromisoformat("2026-10-20T18:30:00+09:00")


def committed_datetime(p):
    rel=str(p.resolve().relative_to(ROOT.resolve()))
    result=subprocess.run(["git","log","--diff-filter=A","--format=%cI","--",rel],
                          cwd=ROOT,capture_output=True,text=True,check=True)
    ts=[s for s in result.stdout.splitlines() if s.strip()]
    if len(ts)!=1:
        raise ValueError(f"Snapshot must have exactly one external ADD commit: {rel}")
    return datetime.fromisoformat(ts[0])


def pmf_nb(mu, alpha, bound=90):
    if not (0<mu<30 and 0<alpha<5):raise ValueError("Invalid μ/NB alpha")
    shape=1.0/alpha
    p=shape/(shape+mu)
    probs=[math.exp(math.lgamma(k+shape)-math.lgamma(shape)-math.lgamma(k+1)+
                    shape*math.log(p)+k*math.log1p(-p)) for k in range(bound+1)]
    mass=sum(probs)
    if mass<.999999:
        raise ValueError(f"Insufficient NB score support mass: {mass}")
    return [x/mass for x in probs]


def score_grid(away_mu,home_mu,alpha):
    a=pmf_nb(away_mu,alpha)
    h=pmf_nb(home_mu,alpha)
    scores={(away,home):pa*ph for away,pa in enumerate(a) for home,ph in enumerate(h)}
    sums={}
    for t in [x+.5 for x in range(2,16)]:
        over=sum(v for (aa,hh),v in scores.items() if aa+hh>t)
        sums[str(t)]={"OVER":over,"UNDER":1-over,"PUSH":0.0}
    outcomes={
        "AWAY_WIN_9IN":sum(v for (aa,hh),v in scores.items() if aa>hh),
        "DRAW_AFTER_9IN":sum(v for (aa,hh),v in scores.items() if aa==hh),
        "HOME_WIN_9IN":sum(v for (aa,hh),v in scores.items() if aa<hh),
    }
    win1loss={
        "AWAY_2PLUS":sum(v for (aa,hh),v in scores.items() if aa>=hh+2),
        "WITHIN_1":sum(v for (aa,hh),v in scores.items() if abs(aa-hh)<=1),
        "HOME_2PLUS":sum(v for (aa,hh),v in scores.items() if hh>=aa+2),
    }
    assert abs(sum(outcomes.values())-1)<1e-9
    assert abs(sum(win1loss.values())-1)<1e-9
    return {"three_way_after_nine_innings":outcomes,
            "baseball_win_one_loss_after_nine":win1loss,
            "total_run_half_line_grid":sums,
            "mean_away_runs":away_mu,"mean_home_runs":home_mu,"nb_alpha":alpha}


def load():
    candidates=sorted(INPUT_DIR.glob("2026-10-20_*.json"))
    if len(candidates)!=1:
        raise ValueError(f"Require exactly one first prospective snapshot; got {len(candidates)}")
    p=candidates[0]
    snap=json.loads(p.read_text(encoding="utf-8"))
    if snap.get("capture_type")!="PROSPECTIVE_CLEAN_PRE_RESEARCH":
        raise ValueError("not original market-unseen prospective snapshot")
    if snap.get("market_state",{}).get("prices_inspected") is not False:
        raise ValueError("market seen or declaration absent")
    if snap.get("target_date") not in (None,"2026-10-20") and snap.get("challenger_output",{}).get("target_date")!="2026-10-20":
        raise ValueError("snapshot date mismatch")
    if snap.get("challenger_output",{}).get("target_date")!="2026-10-20":
        raise ValueError("wrong challenger output date")
    created=datetime.fromisoformat(snap["captured_at_utc"].replace("Z","+00:00"))
    first_commit=committed_datetime(p)
    if created>=KICKOFF or first_commit>=KICKOFF or first_commit<created:
        raise ValueError("CLEAN_PRE commit must follow capture and predate 10/20 first pitch")
    games=snap["challenger_output"]["games"]
    if len(games)!=1:
        raise ValueError("exactly one matchup required")
    game=games[0]
    if game["away_team"]!="HT" or game["home_team"]!="LG":
        raise ValueError("unexpected matchup")
    if game["game_pk"]!="33331020HTLG02026":
        raise ValueError("unexpected postseason game id")
    mu=game["mechanical_mu"]
    required=("f5_away","f5_home","full_away","full_home")
    if not all(key in mu and isinstance(mu[key],(float,int)) and math.isfinite(mu[key]) and 0<mu[key]<20 for key in required):
        raise ValueError("missing/bad model μ")
    fixed=json.loads(MODEL.read_text(encoding="utf-8"))
    return p,snap,mu,fixed,created,first_commit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=OUTPUT)
    ap.add_argument("--allow-existing",action="store_true")
    args=ap.parse_args()
    if args.out.exists() and not args.allow_existing:
        raise SystemExit("Refusing overwrite of original market-unseen probability LOCK")
    p,snap,mu,model,created,commit=load()
    now=datetime.now(KICKOFF.tzinfo)
    if now>=KICKOFF:raise ValueError("Cannot newly derive distribution after first pitch")
    f5=score_grid(mu["f5_away"],mu["f5_home"],float(model["alphas"]["f5"]))
    full=score_grid(mu["full_away"],mu["full_home"],float(model["alphas"]["full"]))
    obj={
        "status":"PREGAME_GENERATED_NEEDS_GITHUB_COMMIT_BEFORE_KICKOFF",
        "date":"2026-10-20","game":"HT@LG","game_pk":"33331020HTLG02026",
        "created_at_kst":now.isoformat(),"model_type":"MU_V1_2_RECOVERED_CHALLENGER_FIXED_9IN_NEGATIVE_BINOMIAL_INDEPENDENCE",
        "model_version":snap["challenger_version"],
        "model_input_snapshot":str(p.relative_to(ROOT)),
        "model_input_snapshot_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "original_snapshot_created_utc":created.isoformat(),
        "original_snapshot_first_git_add_utc":commit.isoformat(),
        "market_state":"NO_TARGET_GAME_BOOK_PRICES_USED",
        "fixed_nb_model_spec":"research/mu-v1.2/model_spec.json",
        "F5":f5,"full_9_in":full,
        "no_baseball_extra_innings_model":True,
        "lineup_adjustments":False,
        "retrospective_betting_roi_certified":False,
        "champion_model":False,
        "no_betting_action":True,
        "warning":"Expected scores use regular-season 2024-26 research rebuild; no 2026 postseason PBP context in mechanical features. Full game includes 9-inning assumptions; actual extras/settlement needs separate validation."
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({
        "status":obj["status"],"game":obj["game"],"mu":mu,
        "F5":f5["three_way_after_nine_innings"],
        "FULL":full["three_way_after_nine_innings"],
        "WIN1LOSS":full["baseball_win_one_loss_after_nine"]},ensure_ascii=False))


if __name__=="__main__":main()

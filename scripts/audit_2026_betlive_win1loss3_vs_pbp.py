#!/usr/bin/env python3
"""Independently score 2026 KBO 3-outcome WIN1LOSE archive against verified PBP.

WIN1LOSE: HOME by >=2; one-run margin including ties; AWAY by >=2.
Official Sports Toto 2023-07-14 announcement:
https://www.sportstoto.co.kr/view.php?idx=2402&type=notice

All bookmaker price timestamps are UNVERIFIED. Reports are descriptive
probability calibration, never a clean-pre return simulation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd

MARKET_ID="FULL_WIN1LOSS_3WAY"
OPTIONS=("HOME_2PLUS","WITHIN_1","AWAY_2PLUS")
BOOK=Path("research/market_fair_odds/betlive_2026_r40_r104_win1loss3_archived.csv")
SOURCE_AUDIT=Path("research/market_fair_odds/betlive_2026_r40_r104_win1loss3_source_audit.json")
JOINED=Path("research/market_fair_odds/betlive_2026_r40_r104_win1loss3_pbp_joined.csv")
RESULT=Path("research/market_fair_odds/betlive_2026_r40_r104_win1loss3_calibration.json")


def actual_event(home:int,away:int)->str:
    delta=home-away
    if delta>=2:return "HOME_2PLUS"
    if delta<=-2:return "AWAY_2PLUS"
    return "WITHIN_1"


def fair(book:dict[str,float])->tuple[dict[str,float],float]:
    if set(book)!=set(OPTIONS) or any(x<=1.0 or not math.isfinite(x) for x in book.values()):
        raise ValueError("Invalid 3-way book")
    inverse={k:1.0/v for k,v in book.items()}
    z=sum(inverse.values())
    return {k:v/z for k,v in inverse.items()},z-1.0


def get_verified_games(pbp_path:Path,status_path:Path,ids:set[str]):
    digest=hashlib.sha256(pbp_path.read_bytes()).hexdigest()
    status=json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("validation")!="PASS" or status.get("sha256_parquet")!=digest:
        raise ValueError("PBP status not PASS or SHA doesn't match published release")
    df=pd.read_parquet(pbp_path)
    required={"game_pk","at_bat_number","pitch_number","home_team",
              "away_team","post_home_score","post_away_score"}
    missing=required-set(df.columns)
    if missing:raise ValueError(f"PBP missing required fields: {sorted(missing)}")
    df=df.loc[df["game_pk"].astype(str).isin(ids)].copy()
    df["_ab"]=pd.to_numeric(df["at_bat_number"],errors="coerce")
    df["_pitch"]=pd.to_numeric(df["pitch_number"],errors="coerce")
    df["_row"]=range(len(df))
    if df[["_ab","_pitch","post_home_score","post_away_score"]].isna().any().any():
        raise ValueError("PBP contains non-final/missing key or score for target games")
    df=df.sort_values(["game_pk","_ab","_pitch","_row"],kind="stable")
    ends=df.groupby("game_pk",sort=False).tail(1)
    out={}
    for r in ends.itertuples(index=False):
        game=str(r.game_pk)
        out[game]=dict(home=int(r.post_home_score),away=int(r.post_away_score),
                       ht=str(r.home_team),at=str(r.away_team))
    return out,digest,str(status.get("last_success_date"))


def join(rows:list[dict],pbp_games:dict):
    enriched=[];issues=[]
    for r in rows:
        gid=r["game_pk"]
        actual=pbp_games.get(gid)
        if not actual:
            issues.append({"game_pk":gid,"issue":"PBP_NOT_FOUND"})
            continue
        if actual["ht"]!=r["home_team"] or actual["at"]!=r["away_team"]:
            issues.append({"game_pk":gid,"issue":"PBP_HOME_AWAY_MISMATCH",
                           "pbp":actual,"source":(r["home_team"],r["away_team"])})
            continue
        b={
            "HOME_2PLUS":float(r["win_2plus_odds"]),
            "WITHIN_1":float(r["within_1_odds"]),
            "AWAY_2PLUS":float(r["lose_2plus_odds"]),
        }
        probs,overround=fair(b)
        outcome=actual_event(actual["home"],actual["away"])
        if r.get("inferred_outcome_from_pbp"):
            issues.append({"game_pk":gid,"issue":"PRIOR_LABEL_NOT_EMPTY_CONFLICT"})
            continue
        scored=dict(r)
        scored.update({
            "actual_home_score":actual["home"],
            "actual_away_score":actual["away"],
            "verified_outcome":outcome,
            "fair_home_2plus":probs["HOME_2PLUS"],
            "fair_within_1":probs["WITHIN_1"],
            "fair_away_2plus":probs["AWAY_2PLUS"],
            "book_overround":overround,
            "outcome_source":"SHA_VERIFIED_2026_PBP",
        })
        enriched.append(scored)
    return enriched,issues


def percentile_bootstrap(obs:list[dict],reps=5000,seed=20261018):
    rng=random.Random(seed)
    keys=list(range(len(obs)))
    results=[]
    for _ in range(reps):
        sample=[obs[rng.choice(keys)] for _ in keys]
        results.append(sum(z["favorite_hit"]-z["favorite_p"] for z in sample)/len(sample))
    results.sort()
    return [results[int((reps-1)*.025)],results[int((reps-1)*.975)]]


def reliable_bins(points, width=.1):
    by=defaultdict(list)
    for x in points:
        k=min(9,int(x["p"]/width))
        by[k].append(x)
    result=[]
    for k,items in sorted(by.items()):
        n=len(items)
        p=sum(v["p"] for v in items)/n
        actual=sum(v["hit"] for v in items)/n
        result.append({"bin":f"{k/10:.1f}-{(k+1)/10:.1f}",
                       "n":n,"mean_probability":p,"observed_frequency":actual,
                       "calibration_error":actual-p})
    return result


def analyze(enriched:list[dict], source_audit:dict,sha:str,through:str,issues):
    if not enriched:raise ValueError("No verified bookmaker and PBP paired games")
    obs=[]
    for r in enriched:
        prob={k:float(r[{"HOME_2PLUS":"fair_home_2plus",
                          "WITHIN_1":"fair_within_1",
                          "AWAY_2PLUS":"fair_away_2plus"}[k]]) for k in OPTIONS}
        win=r["verified_outcome"]
        fav=sorted(OPTIONS,key=lambda k:(-prob[k],k))[0]
        prices={"HOME_2PLUS":float(r["win_2plus_odds"]),
                "WITHIN_1":float(r["within_1_odds"]),
                "AWAY_2PLUS":float(r["lose_2plus_odds"])}
        obs.append({
            "game_pk":r["game_pk"],"date":r["game_date"],
            "outcome":win,"probs":prob,"price":prices,"favorite":fav,
            "favorite_p":prob[fav],"favorite_hit":int(win==fav),
            "brier":sum((prob[k]-int(win==k))**2 for k in OPTIONS),
            "logloss":-math.log(max(prob[win],1e-12)),
            "favorite_unit_profit_if_quote_had_been_executable":
                prices[fav]-1.0 if win==fav else -1.0,
            "overround":float(r["book_overround"]),
        })
    n=len(obs)
    events=Counter(x["outcome"] for x in obs)
    class_prob={k:sum(z["probs"][k] for z in obs)/n for k in OPTIONS}
    class_actual={k:events[k]/n for k in OPTIONS}
    class_cal={}
    for k in OPTIONS:
        values=[{"p":z["probs"][k],"hit":int(z["outcome"]==k)} for z in obs]
        bins=reliable_bins(values)
        class_cal[k]={"mean_no_vig":class_prob[k],
                      "observed_rate":class_actual[k],
                      "observed_minus_expected":class_actual[k]-class_prob[k],
                      "ece":sum(abs(b["calibration_error"])*b["n"] for b in bins)/n,
                      "reliability":bins}
    months=defaultdict(list)
    for x in obs:months[x["date"][:7]].append(x)
    per_month=[]
    for month,x in sorted(months.items()):
        per_month.append({"month":month,"n":len(x),
                          "mean_favorite_probability":sum(z["favorite_p"] for z in x)/len(x),
                          "observed_favorite_frequency":sum(z["favorite_hit"] for z in x)/len(x),
                          "brier":sum(z["brier"] for z in x)/len(x)})
    report={
        "status":"HISTORICAL_3WAY_NO_VIG_CALIBRATION_DESCRIPTIVE_ONLY",
        "training_data_not_used":"NO_FORECAST_FITTING_PERFORMED",
        "market":"KBO FULL_WIN1LOSS_3WAY (HOME 2+, <=1 margin, AWAY 2+)",
        "official_rules":"https://www.sportstoto.co.kr/view.php?idx=2402&type=notice",
        "market_label_warning":"BETLIVE columns 승/무/패; KBO middle is 1-run-or-less INCLUDING 0, not tie-only",
        "source":"https://betlive.kr/proto/260NNN archive",
        "source_quote_time":"HISTORICAL_POSTGAME_PAGE_NO_SOURCE_PRE_START_TIMESTAMP",
        "source_coverage_rounds":[source_audit["first_round"],source_audit["last_round"]],
        "total_archived_rows":int(source_audit["distinct_game_markets"]),
        "independent_pbp_joined":n,
        "nonmatched_or_conflicted_rows":len(issues),
        "nonmatched_details":issues[:140],
        "cross_round_duplicates":source_audit["cross_round_duplicates"],
        "source_robots_compliance":"CHECKED_BY_COLLECTOR",
        "source_result_PBP_sha256":sha,"source_result_PBP_through":through,
        "all_source_rows_checked_against_pbp":n==int(source_audit["distinct_game_markets"]) and not issues,
        "mean_overround":sum(z["overround"] for z in obs)/n,
        "observed_class_counts":{k:events[k] for k in OPTIONS},
        "mean_no_vig_class_probabilities":class_prob,
        "observed_class_rates":class_actual,
        "class_calibration":class_cal,
        "favorite_mean_no_vig_probability":sum(z["favorite_p"] for z in obs)/n,
        "favorite_actual_hit_rate":sum(z["favorite_hit"] for z in obs)/n,
        "favorite_calibration_gap":sum(z["favorite_hit"]-z["favorite_p"] for z in obs)/n,
        "favorite_calibration_gap_bootstrap95_game":[*percentile_bootstrap(obs)] if n>15 else None,
        "multiclass_brier":sum(z["brier"] for z in obs)/n,
        "multiclass_logloss":sum(z["logloss"] for z in obs)/n,
        "hypothetical_favorite_roi_if_archived_price_executable":
            sum(z["favorite_unit_profit_if_quote_had_been_executable"] for z in obs)/n,
        "by_game_month":per_month,
        "never_conclude_positive_EV_from_archival_quote":True,
        "no_CLEAN_PRE_model_compared":True,
        "not_identical_to_TWO_WAY_FULL_ML2":True,
        "limitations":[
            "BETLIVE does not show the time each posted price was executable; return is entirely hypothetical",
            "3-way WIN1LOSS not 2-way moneyline; do not merge probability bins across markets",
            "Source selection and season/date coverage are nonrandom; no full-season claims without exhaustive verification",
            "Missing PBP games excluded, never assigned prices or labels",
            "Without verified pregame model LOCK, one cannot claim a model beat the market"
        ],
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--book",type=Path,default=BOOK)
    p.add_argument("--source-audit",type=Path,default=SOURCE_AUDIT)
    p.add_argument("--pbp",type=Path,required=True)
    p.add_argument("--status",type=Path,default=Path("state/status.json"))
    p.add_argument("--joined",type=Path,default=JOINED)
    p.add_argument("--output",type=Path,default=RESULT)
    x=p.parse_args()
    source=json.loads(x.source_audit.read_text(encoding="utf-8"))
    if source["status"]!="HISTORICAL_3WAY_QUOTES_NO_PREGAME_TIMESTAMP":
        raise ValueError("No verified source provenance")
    with x.book.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=source["distinct_game_markets"]:raise ValueError("Audit source row count differs")
    ids=[r["game_pk"] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate odds for same game")
    if not all(r["market_type"]==MARKET_ID and
               r["source_price_temporality"]=="HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED"
               for r in rows):
        raise ValueError("Odds are not uniformly historical three-way")
    games,sha,through=get_verified_games(x.pbp,x.status,set(ids))
    joined,issues=join(rows,games)
    report=analyze(joined,source,sha,through,issues)
    x.joined.parent.mkdir(parents=True,exist_ok=True)
    if joined:
        with x.joined.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=joined[0].keys());w.writeheader();w.writerows(joined)
    x.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in (
        "status","independent_pbp_joined","nonmatched_or_conflicted_rows",
        "mean_overround","multiclass_brier","favorite_mean_no_vig_probability",
        "favorite_actual_hit_rate","favorite_calibration_gap",
        "hypothetical_favorite_roi_if_archived_price_executable"
    )},ensure_ascii=False))
    if issues or n_missing:=len(rows)-len(joined):
        raise SystemExit(f"INCOMPLETE_EXACT_PBP_SCORE_COVERAGE: missing={n_missing}, conflicts={len(issues)}")


if __name__=="__main__":
    main()

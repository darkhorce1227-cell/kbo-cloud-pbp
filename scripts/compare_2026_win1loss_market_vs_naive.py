#!/usr/bin/env python3
"""Benchmark archived KBO WIN1LOSS probabilities vs ex-ante naive 3-way priors.

Market probabilities: original archived odds timestamps UNVERIFIED.
Baselines: fixed uniform 1/3 each, and strictly PAST-DATE-ONLY
Dirichlet-smoothed season-to-date event frequencies with alpha=10
per outcome (fixed in source, never chosen on scored outcomes).

Performance estimates are descriptive market-information studies, NOT
a prefirst-pitch executable price or original model CLEAN PRE benchmark.
"""
from __future__ import annotations

from collections import defaultdict,Counter
from datetime import date
from pathlib import Path
import csv
import json
import math
import random

BASE=Path("research/market_fair_odds")
SOURCE=BASE/"betlive_2026_r40_r122_win1loss3_pbp_joined.csv"
OUTPUT=BASE/"betlive_2026_r40_r122_win1loss3_information_benchmark.json"
CLASS=("HOME_2PLUS","WITHIN_1","AWAY_2PLUS")
ALPHA=10


def prob_score(probs:dict, outcome:str):
    if set(probs)!=set(CLASS) or abs(sum(probs.values())-1)>1e-8:
        raise ValueError("Malformed 3way probabilities")
    if outcome not in CLASS:
        raise ValueError("Invalid recorded game outcome")
    return (sum((probs[k]-int(k==outcome))**2 for k in CLASS),
            -math.log(max(probs[outcome],1e-12)))


def score(records):
    by_day=defaultdict(list)
    for r in records:
        d=date.fromisoformat(r["game_date"])
        if r["market_type"]!="FULL_WIN1LOSS_3WAY" or r["outcome_source"]!="SHA_VERIFIED_2026_PBP":
            raise ValueError("Unverified settlement/market type")
        if r["source_price_temporality"]!="HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED":
            raise ValueError("False or missing archival quote status")
        by_day[d].append(r)
    assert len(by_day)>15
    already=Counter()
    results=[]
    for d in sorted(by_day):
        # Freeze the prior for the *entire* game date BEFORE scoring outcomes
        # or incorporating same-day completed scores.
        past_n=sum(already.values())
        current_p={k:(already[k]+ALPHA)/(past_n+ALPHA*len(CLASS)) for k in CLASS}
        uniform={k:1/len(CLASS) for k in CLASS}
        for r in by_day[d]:
            mp={"HOME_2PLUS":float(r["fair_home_2plus"]),
                "WITHIN_1":float(r["fair_within_1"]),
                "AWAY_2PLUS":float(r["fair_away_2plus"])}
            win=r["verified_outcome"]
            mb,ml=prob_score(mp,win)
            pb,pl=prob_score(current_p,win)
            ub,ul=prob_score(uniform,win)
            results.append({
               "game_pk":r["game_pk"],"date":d.isoformat(),
               "week_key":f"{d.isocalendar().year}-W{d.isocalendar().week:02d}",
               "actual":win,
               "predicted_market":mp,"predicted_before_day_climatology":current_p,
               "climatology_prior_data_n":past_n,
               "market_brier":mb,"past_only_brier":pb,"uniform_brier":ub,
               "market_logloss":ml,"past_only_logloss":pl,"uniform_logloss":ul,
               "paired_brier_improvement_market_over_past_only":pb-mb,
               "paired_brier_improvement_market_over_uniform":ub-mb,
            })
        already.update(r["verified_outcome"] for r in by_day[d])
    return results


def grouped_bootstrap(rows, data_key, reps=5000, seed=20261018):
    groups=defaultdict(list)
    for row in rows:groups[row["week_key"]].append(row[data_key])
    keys=list(groups)
    if len(keys)<8:raise ValueError("Too few calendar-week clusters")
    rng=random.Random(seed)
    vals=[]
    for _ in range(reps):
        select=[v for _ in keys for v in groups[rng.choice(keys)]]
        vals.append(sum(select)/len(select))
    vals.sort()
    return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]


def main():
    with SOURCE.open(encoding="utf-8",newline="") as f:
        original=list(csv.DictReader(f))
    if not original:raise ValueError("Historical KBO PBP scored odds unavailable")
    if len({r["game_pk"] for r in original})!=len(original):
        raise ValueError("Multiple price snapshots for same game")
    rows=score(original)
    if len(rows)!=len(original):raise ValueError("Scoring coverage mismatch")
    n=len(rows)
    avg=lambda k:sum(r[k] for r in rows)/n
    report={
        "status":"DESCRIPTIVE_MARKET_INFORMATION_BENCHMARK_NON_EXECUTABLE",
        "market":"KBO FULL_WIN1LOSS_3WAY",
        "game_count":n,
        "weeks_bootstrap_groups":len({r["week_key"] for r in rows}),
        "source_price_timestamp":"UNVERIFIED",
        "source_pbp_score":"SHA_VERIFIED_2026_PBP",
        "baseline_uniform":"Fixed 1/3 for HOME_2PLUS/WITHIN_1/AWAY_2PLUS",
        "baseline_past_only":f"Dirichlet pseudo-count {ALPHA} PER class, no target or same-date outcome used",
        "first_game_date":min(r["date"] for r in rows),
        "last_game_date":max(r["date"] for r in rows),
        "market_multiclass_brier":avg("market_brier"),
        "past_only_climatology_brier":avg("past_only_brier"),
        "uniform_brier":avg("uniform_brier"),
        "market_logloss":avg("market_logloss"),
        "past_only_climatology_logloss":avg("past_only_logloss"),
        "uniform_logloss":avg("uniform_logloss"),
        "paired_brier_improvement_market_over_past_only":avg(
            "paired_brier_improvement_market_over_past_only"),
        "paired_brier_improvement_market_over_uniform":avg(
            "paired_brier_improvement_market_over_uniform"),
        "paired_brier_improvement_market_over_past_only_95CI_week_cluster":
            grouped_bootstrap(rows,"paired_brier_improvement_market_over_past_only"),
        "paired_brier_improvement_market_over_uniform_95CI_week_cluster":
            grouped_bootstrap(rows,"paired_brier_improvement_market_over_uniform"),
        "market_brier_skill_vs_past_only":
            1-avg("market_brier")/avg("past_only_brier"),
        "market_brier_skill_vs_uniform":1-avg("market_brier")/avg("uniform_brier"),
        "no_model_predictions_ever_used":True,
        "no_real_betting_roi_inferred":True,
        "limitations":[
            "Data are selected 2026 public bookmaker archive games; historical page quoted prices cannot be authenticated as available before first pitch",
            "Market's descriptive probability information advantage is not proof of trading/portfolio edge after bookmaker margin",
            "Past-only baseline uses class frequency from earlier game dates only; within-day outcomes are never observed by the baseline",
            "Fixed Dirichlet smoothing alpha=10 is an arbitrary, prespecified baseline, not optimized on this sample",
            "Calendar-week cluster bootstrap estimates uncertainty across this selected sample, not full-season generalization"
        ],
    }
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in (
        "game_count","market_multiclass_brier","past_only_climatology_brier",
        "uniform_brier","paired_brier_improvement_market_over_past_only",
        "paired_brier_improvement_market_over_past_only_95CI_week_cluster",
        "market_brier_skill_vs_past_only")},ensure_ascii=False))


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Descriptive market fair-odds calibration of archival Proto KBO records.

No ex-ante CLEAN PRE or verified quote timestamp exists here. No-vig prices are
conditional on a settled outcome (2-way refunded draws omitted). Unit ROI at
displayed odds is hypothetical and explicitly NOT a verified executable edge.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
import random
from pathlib import Path

MARKETS=("FULL_ML2","FULL_HC2","FULL_OU")


def brier(p, win):
    return sum((v-int(k==win))**2 for k,v in p.items())


def logloss(p, win):
    return -math.log(max(p[win],1e-12))


def rows_for_calibration(path:Path)->list[dict]:
    with path.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError("Missing archived market data")
    seen=set();good=[]
    for row in rows:
        key=row["game_pk"],row["market_type"],row["line"]
        if key in seen:raise ValueError(f"Duplicate game-market {key}")
        seen.add(key)
        if row["market_type"] not in MARKETS:raise ValueError(f"Wrong market type {key}")
        if row["source_price_temporality"]!="HISTORICAL_PAGE_RETROSPECTIVE_PREMATCH_PRICE_TIME_UNVERIFIED":
            raise ValueError(f"Unexpected quote temporality {key}")
        outcome=row["settled_outcome"]
        status=row["settlement_status"]
        if status not in ("SETTLED","PUSH","VOID"):raise ValueError("Bad settlement state")
        offers=json.loads(row["odds_json"])
        opens=json.loads(row["opening_odds_json"])
        expected={"HOME","AWAY"} if row["market_type"]!="FULL_OU" else {"UNDER","OVER"}
        if set(offers)!=expected or set(opens)!=expected:
            raise ValueError("Incomplete offered outcomes")
        for book in [offers,opens]:
            if not all(isinstance(o,(float,int)) and math.isfinite(o) and o>1 for o in book.values()):
                raise ValueError("Nonphysical decimal book")
        if status=="SETTLED" and outcome not in expected:
            raise ValueError("Settled outcome absent from book")
        if status!="SETTLED" and outcome:
            raise ValueError("Push/VOID cannot have a winning outcome")
        good.append({
            "game":row["game_pk"],"type":row["market_type"],"line":row["line"],
            "settlement":status,"outcome":outcome,"odds":offers,"opening":opens,
            "opening_provenance":row["opening_value_provenance"],
        })
    return good


def no_vig(odds):
    inv={k:1/v for k,v in odds.items()}
    denom=sum(inv.values())
    return {k:v/denom for k,v in inv.items()},denom-1.0


def favorite_record(row,kind):
    offered=row["opening"] if kind=="opening" else row["odds"]
    fair,overround=no_vig(offered)
    # Resolve exact equal odds by stable HOME > AWAY / UNDER > OVER alphabetic.
    favorite=sorted(offered,key=lambda x:(-fair[x],x))[0]
    return {
        "game":row["game"],"type":row["type"],"line":row["line"],"p":fair[favorite],
        "y":float(row["outcome"]==favorite),
        "brier":brier(fair,row["outcome"]),
        "logloss":logloss(fair,row["outcome"]),
        "overround":overround,"win":row["outcome"],
        "favorite":favorite,"favorite_odds":offered[favorite],
        "favorite_unit_net":offered[favorite]-1 if favorite==row["outcome"] else -1,
    }


def reliability(values):
    bins=defaultdict(list)
    for r in values:
        i=min(9,int(r["p"]*10))
        bins[i].append(r)
    out=[]
    for k,group in sorted(bins.items()):
        n=len(group);p=sum(r["p"] for r in group)/n
        y=sum(r["y"] for r in group)/n
        out.append({"range":f"{k/10:.1f}-{(k+1)/10:.1f}",
                    "n":n,"mean_fair_probability":p,"observed_favorite_hit_rate":y,
                    "observed_minus_predicted":y-p})
    return out


def bootstrap_game_favorite_gap(records,reps=3000):
    if not records:return None
    by=defaultdict(list)
    for r in records:by[r["game"]].append(r)
    keys=list(by);rng=random.Random(20261018);stats=[]
    for _ in range(reps):
        drawn=[r for _ in keys for r in by[rng.choice(keys)]]
        stats.append(sum(r["y"]-r["p"] for r in drawn)/len(drawn))
    stats.sort()
    return [stats[int(.025*(reps-1))],stats[int(.975*(reps-1))]]


def summarize(values):
    if not values:return {"n":0}
    n=len(values);out={
        "n":n,"independent_games":len(set(r["game"] for r in values)),
        "mean_fair_favorite_probability":sum(r["p"] for r in values)/n,
        "actual_favorite_hit_rate":sum(r["y"] for r in values)/n,
        "favorite_calibration_gap":sum(r["y"]-r["p"] for r in values)/n,
        "favorite_gap_game_cluster_bootstrap_95":bootstrap_game_favorite_gap(values),
        "multiclass_brier":sum(r["brier"] for r in values)/n,
        "log_loss":sum(r["logloss"] for r in values)/n,
        "average_overround":sum(r["overround"] for r in values)/n,
        "favorite_hypothetical_unit_roi_at_archived_displayed_odds":
            sum(r["favorite_unit_net"] for r in values)/n,
        "favorite_reliability_bins":reliability(values),
    }
    out["ece_favorite_10bins"]=sum(
        b["n"]*abs(b["observed_minus_predicted"]) for b in out["favorite_reliability_bins"]
    )/n
    return out


def analyze(data, result_audit):
    if result_audit.get("status")!="PASS":
        raise ValueError("Independent PBP final result audit is not PASS")
    if result_audit.get("market_unique_games")!=len(set(r["game"] for r in data)):
        raise ValueError("PBP audit matches a different market sample")
    results={}
    for typ in MARKETS:
        relevant=[x for x in data if x["type"]==typ]
        settled=[x for x in relevant if x["settlement"]=="SETTLED"]
        obj={
            "offered_markets":len(relevant),"settled":len(settled),
            "push":sum(x["settlement"]=="PUSH" for x in relevant),
            "void":sum(x["settlement"]=="VOID" for x in relevant),
            "opening_proxy":summarize([favorite_record(x,"opening") for x in settled]),
            "displayed_quote":summarize([favorite_record(x,"display") for x in settled]),
            "opening_equals_display_assumed_count":
              sum(x["opening_provenance"]=="NO_CHANGE_ASSUMED_EQUAL_TO_DISPLAY" for x in relevant),
        }
        results[typ]=obj
    return {
        "status":"DESCRIPTIVE_HISTORICAL_MARKET_CALIBRATION_NOT_CLEAN_PRE",
        "source":"sharpPick retrospective 2026 Proto page archive",
        "timestamp_certainty":"UNVERIFIED_PRICE_TIME",
        "independent_result_audit":"PASS",
        "number_of_unique_games":len(set(x["game"] for x in data)),
        "distinct_game_markets":len(data),
        "markets":results,
        "limitations":[
            "No original pregame odds quote timestamp; displayed-price ROI is hypothetical and may not have been executable",
            "Historical opening prices inferred equal to displayed prices when source shows no movement",
            "Markets conditioned on settled non-refunded outcomes; model comparisons not possible without CLEAN PRE locks",
            "Clustering by baseball game and multiple outcomes/markets matters; marginal 95% gaps not evidence of long-term profits",
            "Outcomes checked against independently SHA-validated KBO PBP; this proves only results, not bettable odds",
            "No betting advice and no automated wagering",
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--book",type=Path,default=Path("research/market_fair_odds/sharppick_2026_proto_r112_r122_kbo_reconstructed.csv"))
    p.add_argument("--result-audit",type=Path,default=Path("research/market_fair_odds/sharppick_2026_r112_r122_pbp_result_audit.json"))
    p.add_argument("--output",type=Path,default=Path("research/market_fair_odds/sharppick_2026_r112_r122_market_calibration.json"))
    args=p.parse_args()
    report=analyze(rows_for_calibration(args.book),json.loads(args.result_audit.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"games":report["number_of_unique_games"],
                      "markets":{k:{"n":v["displayed_quote"]["n"],
                                    "average_overround":v["displayed_quote"].get("average_overround"),
                                    "mean_pred":v["displayed_quote"].get("mean_fair_favorite_probability"),
                                    "actual":v["displayed_quote"].get("actual_favorite_hit_rate"),
                                    "brier":v["displayed_quote"].get("multiclass_brier")}
                                 for k,v in report["markets"].items()}},ensure_ascii=False))


if __name__=="__main__":
    main()

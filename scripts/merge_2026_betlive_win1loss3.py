#!/usr/bin/env python3
"""Strictly merge BETLIVE 2026 KBO 3way books across early and late rounds.

One game, one 3-outcome market. Original round and source provenance retained;
all quotes retrospective with unverifiable pre-first-pitch prices.
"""
from __future__ import annotations

import csv
from pathlib import Path
import json

DIR=Path("research/market_fair_odds")
PARTS=[(40,104),(105,122)]
FROM=40
THROUGH=122


def combine(base:Path=DIR):
    all_rows=[];all_rounds=[];all_issues=[];fields=None
    for a,b in PARTS:
        prefix=f"betlive_2026_r{a}_r{b}_win1loss3"
        audit=json.loads((base/(prefix+"_source_audit.json")).read_text(encoding="utf-8"))
        if audit["status"]!="HISTORICAL_3WAY_QUOTES_NO_PREGAME_TIMESTAMP":
            raise ValueError(f"Source audit invalid {a}-{b}")
        with (base/(prefix+"_archived.csv")).open(encoding="utf-8",newline="") as f:
            reader=csv.DictReader(f);rowlist=list(reader)
            if fields is None:fields=reader.fieldnames
            elif fields!=reader.fieldnames:raise ValueError("Source schema mismatch")
        if len(rowlist)!=audit["distinct_game_markets"]:
            raise ValueError(f"Bad expected source market count {a}-{b}")
        all_rows.extend(rowlist)
        all_rounds.extend(audit["rounds"])
        all_issues.extend(audit["issues"])
    seen={};duplicates=[];price_conflicts=[]
    for row in sorted(all_rows,key=lambda x:(int(x["proto_round"]),x["game_pk"])):
        if row["market_type"]!="FULL_WIN1LOSS_3WAY" or (
            row["source_price_temporality"]!="HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED"
        ):
            raise ValueError("Attempted market/temporality contamination")
        gid=row["game_pk"]
        if gid in seen:
            previous=seen[gid]
            duplicates.append({"game_pk":gid,"first_round":previous["proto_round"],
                               "later_round":row["proto_round"]})
            odds=lambda z:(z["win_2plus_odds"],z["within_1_odds"],z["lose_2plus_odds"])
            if odds(previous)!=odds(row):
                price_conflicts.append({"game_pk":gid,
                      "first_round":previous["proto_round"],"later_round":row["proto_round"],
                      "first_odds":odds(previous),"later_odds":odds(row)})
            continue
        seen[gid]=row
    report={
        "status":"HISTORICAL_3WAY_QUOTES_NO_PREGAME_TIMESTAMP",
        "first_round":FROM,"last_round":THROUGH,
        "source_components":PARTS,
        "rounds":all_rounds,
        "rounds_parsed":sum(r["status"]=="PARSED" for r in all_rounds),
        "rounds_404":[r["round"] for r in all_rounds if r["status"]=="NOT_FOUND_404"],
        "rounds_failed":[r for r in all_rounds if r["status"] not in ("PARSED","NOT_FOUND_404")],
        "source_rows_strict_game_pk_join":len(all_rows),
        "distinct_game_markets":len(seen),
        "cross_round_duplicates":len(duplicates),
        "cross_round_quote_disagreements":price_conflicts[:100],
        "issues_count":sum(r.get("issues",0) for r in all_rounds),
        "issues_from_component_audits_sample":all_issues[:100],
        "market_id":"FULL_WIN1LOSS_3WAY",
        "market_outcomes":{"HOME_2PLUS":"home run diff >= 2",
          "WITHIN_1":"absolute run diff <=1 including ties",
          "AWAY_2PLUS":"away run diff >= 2"},
        "three_way_middle_official_rule_url":
            "https://www.sportstoto.co.kr/view.php?idx=2402&type=notice",
        "must_independently_verify_scores_with_pbp":True,
        "historical_odds_executable_at_first_pitch_unverified":True,
        "never_join_as_regular_ml2":True,
        "not_genuine_prospective_bet_roi":True,
    }
    if report["rounds_failed"]:raise ValueError("Unexpected HTTP/schema failures in component sources")
    return sorted(seen.values(),key=lambda z:(z["game_date"],z["game_pk"])),fields,report


def main():
    rows,fields,report=combine()
    prefix=f"betlive_2026_r{FROM}_r{THROUGH}_win1loss3"
    with (DIR/(prefix+"_archived.csv")).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (DIR/(prefix+"_source_audit.json")).write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"markets":len(rows),"rounds_parsed":report["rounds_parsed"],
                      "missing_rounds":report["rounds_404"],
                      "duplicates":report["cross_round_duplicates"]},ensure_ascii=False))
    if len(rows)<380:
        raise SystemExit("Insufficient combined genuine source rows")


if __name__=="__main__":
    main()

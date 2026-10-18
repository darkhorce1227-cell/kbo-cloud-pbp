#!/usr/bin/env python3
"""Merge two sourced 2026 KBO Proto archival quote collections.

Deduplicates identical game/market/line across archived rounds while
preserving source page URL, original round, odds and explicit 404 gaps.
Refuses conflicting final scores; never fills missing prices or outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

BASE=Path("research/market_fair_odds")
SOURCES=[(101,111),(112,122)]


def merge(directory:Path):
    all_rows=[];source_stats=[]
    for first,last in SOURCES:
        p=directory/f"sharppick_2026_proto_r{first}_r{last}_kbo_reconstructed.csv"
        a=directory/f"sharppick_2026_proto_r{first}_r{last}_audit.json"
        audit=json.loads(a.read_text(encoding="utf-8"))
        if audit["status"]!="SOURCE_RECONSTRUCTED_UNVERIFIED_PRICE_TIMESTAMP":
            raise ValueError(f"Bad source audit status {a}")
        with p.open(encoding="utf-8",newline="") as f:
            read=csv.DictReader(f)
            cols=read.fieldnames
            rows=list(read)
        if len(rows)!=audit["deduplicated_markets"]:
            raise ValueError("CSV does not match source audit")
        if any(not row["source_price_temporality"].startswith("HISTORICAL_PAGE_RETROSPECTIVE_")
               for row in rows):
            raise ValueError("Found a purported live odds quote in retrospective book")
        all_rows.extend(rows)
        source_stats.append({"first":first,"last":last,"source_markets":len(rows),
                             "source_games":audit["unique_games"],
                             "http_404_or_errors":[z["round"] for z in audit["rounds"]
                                                  if z.get("http")!="OK"],
                             "successful_rounds":[z["round"] for z in audit["rounds"]
                                                  if z.get("http")=="OK"]})
    scores={};merged={};cross_round_duplicates=0
    for row in sorted(all_rows,key=lambda x:(int(x["proto_round"]),x["game_pk"],x["market_type"],x["line"])):
        game=row["game_pk"]
        result=(row["home_score"],row["away_score"],row["home_team"],row["away_team"])
        if game in scores and scores[game]!=result:
            raise ValueError(f"Conflicting game outcome after merge: {game}")
        scores[game]=result
        key=game,row["market_type"],row["line"]
        if key in merged:
            cross_round_duplicates+=1
            continue
        merged[key]=row
    by_type=defaultdict(int)
    for r in merged.values():by_type[r["market_type"]]+=1
    report={
        "status":"PASS_RECONSTRUCTED_ARCHIVE_NO_PREMATCH_TIMESTAMP",
        "year":2026,"source_coverage":source_stats,
        "all_rows_before_cross_range_dedup":len(all_rows),
        "duplicate_market_lines_across_ranges":cross_round_duplicates,
        "unique_games":len(scores),"distinct_game_markets":len(merged),
        "by_market_type":dict(by_type),
        "game_score_conflicts":0,
        "quote_timestamp_verified":False,
        "market_book_is_prospective":False,
        "limitations":[
           "Original quote timestamp unverified; no CLEAN PRE model comparisons",
           "Some earlier 2026 rounds absent from public archive; do not label full season",
           "Extra source round observations of the same game are not independent trials",
           "Book outcomes require separate Naver/PBP final score audit"
        ]
    }
    return list(merged.values()),cols,report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output-dir",type=Path,default=BASE)
    args=a.parse_args()
    rows,cols,report=merge(args.output_dir)
    output=args.output_dir/"sharppick_2026_proto_r105_r122_combined_kbo.csv"
    audit=args.output_dir/"sharppick_2026_proto_r105_r122_combined_audit.json"
    with output.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        w.writerows(sorted(rows,key=lambda r:(r["game_date"],r["game_pk"],r["market_type"],r["line"])))
    audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    if len(rows)<300 or len(set(r["game_pk"] for r in rows))<100:
        raise SystemExit("Not enough authentic market records for expanded study")


if __name__=="__main__":
    main()

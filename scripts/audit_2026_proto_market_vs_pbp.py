#!/usr/bin/env python3
"""Score-level audit: historical KBO Proto archive against independently validated Naver PBP.

Never infer missing outcome from betting source; FAIL if any KBO score differs
from full-game final scores reconstructed from the 2026 pitch-level dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(quotes_path: Path, pbp_path: Path, status_path: Path)->dict:
    with quotes_path.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError("No market quotes")
    grouped={}
    for r in rows:
        gid=r["game_pk"]
        val=(int(r["home_score"]),int(r["away_score"]),r["home_team"],r["away_team"])
        if gid in grouped and val!=grouped[gid]:raise ValueError(f"Market scores disagree within game {gid}")
        grouped[gid]=val
    raw=pbp_path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    status=json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("validation")!="PASS" or status.get("sha256_parquet")!=digest:
        raise ValueError("PBP release does not match validated dataset status SHA")
    df=pd.read_parquet(pbp_path)
    required={"game_pk","at_bat_number","pitch_number","home_team","away_team",
              "post_home_score","post_away_score"}
    if missing:=required-set(df.columns):raise ValueError(f"PBP missing {sorted(missing)}")
    df=df[df["game_pk"].astype(str).isin(grouped)].copy()
    if df.empty:raise ValueError("No matching PBP game rows")
    df["_ab"]=pd.to_numeric(df["at_bat_number"],errors="coerce")
    df["_pitch"]=pd.to_numeric(df["pitch_number"],errors="coerce")
    df["_row"]=range(len(df))
    if df[["_ab","_pitch"]].isna().any().any():
        raise ValueError("Null pitch-level keys; cannot establish final score")
    df=df.sort_values(["game_pk","_ab","_pitch","_row"],kind="stable")
    end=df.groupby("game_pk",sort=False).tail(1)
    found={}
    for r in end.itertuples(index=False):
        gid=str(r.game_pk)
        found[gid]=(int(r.post_home_score),int(r.post_away_score),
                    str(r.home_team),str(r.away_team))
    absent=sorted(set(grouped)-set(found))
    mismatches=[
        {"game_pk":gid,"source_result":grouped[gid],"pbp_final":found.get(gid)}
        for gid in grouped if gid in found and grouped[gid]!=found[gid]
    ]
    report={
        "status":"PASS" if not absent and not mismatches else "FAIL",
        "market_unique_games":len(grouped),
        "pbp_found_games":len(found),
        "exact_score_matches":sum(grouped[gid]==found.get(gid) for gid in grouped),
        "missing_pbp_games":absent,
        "score_disagreements":mismatches,
        "pbp_sha256":digest,
        "pbp_status_validation":status["validation"],
        "pbp_status_through_date":status["last_success_date"],
        "market_source_retrospective_only":True,
        "pre_first_pitch_price_timestamp_verified":False,
        "limitation":"Independent Naver PBP result verification is not proof that book's displayed price was executable at first pitch.",
    }
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quotes",type=Path,default=Path("research/market_fair_odds/sharppick_2026_proto_r112_r122_kbo_reconstructed.csv"))
    ap.add_argument("--pbp",type=Path,required=True)
    ap.add_argument("--status",type=Path,default=Path("state/status.json"))
    ap.add_argument("--output",type=Path,default=Path("research/market_fair_odds/sharppick_2026_r112_r122_pbp_result_audit.json"))
    a=ap.parse_args()
    report=audit(a.quotes,a.pbp,a.status)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":
    main()

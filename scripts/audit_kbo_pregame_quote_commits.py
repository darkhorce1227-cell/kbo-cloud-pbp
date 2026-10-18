#!/usr/bin/env python3
"""Check first Git ADD commit of public KBO quote against source-scheduled kickoff.

Even PASS only means a *displayed source quote* was observed and committed
before scheduled first pitch. Real sportsbook orderability, actual first pitch,
and independent CLEAN PRE model lock must be separately established.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from capture_kbo_proto_pregame_quotes import _stable_fingerprint

ROOT=Path(__file__).resolve().parents[1]
QUOTES=ROOT/"state/pregame_market_quotes"
REPORT=ROOT/"research/market_fair_odds/live_quote_capture_audit.json"
REQUIRED={"market_id","scheduled_first_pitch_kst","first_observed_at_utc",
          "decimal_odds","market_line","market_type","home_team","away_team",
          "source_round_url","source_match_url","snapshot_fingerprint_sha256",
          "source_round_response_sha256","displayed_price_is_executable_at_betman",
          "time_verified","future_game_pk","is_clean_pre_model_prediction"}


def timestamp(value):
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timezone missing from evidence")
    return dt.astimezone(timezone.utc)


def original_add_commit(path,root=ROOT):
    rel=str(path.resolve().relative_to(root.resolve()))
    proc=subprocess.run(["git","log","--diff-filter=A","--format=%cI",
                         "--reverse","--",rel],cwd=root,text=True,
                        capture_output=True,check=True)
    lines=[s.strip() for s in proc.stdout.splitlines() if s.strip()]
    if len(lines)>1:raise ValueError("historical quote file re-added multiple times")
    return timestamp(lines[0]) if lines else None


def audit_one(path,root=ROOT,commit_time=None):
    rel=str(path.resolve().relative_to(root.resolve()))
    item=json.loads(path.read_text(encoding="utf-8"))
    if missing:=REQUIRED-set(item):
        raise ValueError(f"missing quote fields: {sorted(missing)}")
    if item["displayed_price_is_executable_at_betman"] is not False:
        raise ValueError("falsely marked as executable official Betman quote")
    if item["is_clean_pre_model_prediction"] is not False or item["future_game_pk"] is not None:
        raise ValueError("falsely marked market odds as CLEAN PRE / game_pk")
    if item["time_verified"]!="OBSERVATION_TIME_ONLY_NOT_BOOK_OPENING_TIME":
        raise ValueError("unverified publisher opening time")
    if item.get("original_prospective_label")!="PUBLIC_PRICE_CAPTURE_REQUIRES_FIRST_COMMIT_TIME_GATE":
        raise ValueError("original pregame quote agreement absent")
    types={"FULL_ML2":{"HOME","AWAY"},"FULL_HC2":{"HOME","AWAY"},
           "FULL_OU":{"UNDER","OVER"}}
    if item["market_type"] not in types:raise ValueError("unknown product")
    prices=item["decimal_odds"]
    if set(prices)!=types[item["market_type"]] or any(not 1<float(v)<1000 for v in prices.values()):
        raise ValueError("bad/incomplete price book")
    if item["snapshot_fingerprint_sha256"]!=_stable_fingerprint(item):
        raise ValueError("immutable quote fingerprint mismatch")
    kickoff=timestamp(item["scheduled_first_pitch_kst"])
    observed=timestamp(item["first_observed_at_utc"])
    commit=original_add_commit(path,root) if commit_time is None else commit_time
    if commit is None:state="NO_EXTERNAL_GIT_ADD_COMMIT"
    elif observed>=kickoff:state="REJECT_LATE_CAPTURE"
    elif commit>=kickoff:state="REJECT_LATE_GIT_COMMIT"
    elif commit<observed:state="REJECT_COMMIT_CLOCK_BEFORE_HTTP_RESPONSE"
    else:state="PASS_PUBLIC_QUOTE_GIT_LOCK_BEFORE_SCHEDULED_FIRST_PITCH"
    return {"source_file":rel,"market_id":item["market_id"],
            "market_type":item["market_type"],
            "observation_utc":observed.isoformat(),
            "first_git_add_commit_utc":commit.isoformat() if commit else None,
            "scheduled_first_pitch_utc":kickoff.isoformat(),"status":state,
            "public_quote_provenance_eligible":state.startswith("PASS_"),
            "confirmed_bettable_before_kickoff":False,
            "genuine_clean_pre_model":False,
            "original_live_market_price_is_same_as_official_protobetman":"NOT_VERIFIED"}


def collect(quote_dir=QUOTES,root=ROOT):
    rows=[]
    for p in sorted(quote_dir.rglob("*.json")) if quote_dir.exists() else []:
        try:rows.append(audit_one(p,root))
        except (ValueError,OSError,json.JSONDecodeError) as e:
            rows.append({"source_file":str(p),"status":"REJECT_BAD_SOURCE_EVIDENCE",
                         "detail":str(e)[:230],"public_quote_provenance_eligible":False})
    return {"status":"AUDITED_MARKET_QUOTE_GIT_PROVENANCE_ONLY",
            "total_quote_files":len(rows),
            "public_pre_scheduled_start_git_provenance_pass":
                sum(x["public_quote_provenance_eligible"] for x in rows),
            "not_an_executable_sportsbook_quote":True,
            "cannot_be_scored_as_clean_pre_model":True,
            "actual_kbo_first_pitch_must_be_verified_independently":True,
            "completed_model_vs_market_pairs":0,
            "quote_audits":rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quote-dir",type=Path,default=QUOTES)
    ap.add_argument("--output",type=Path,default=REPORT)
    ap.add_argument("--strict",action="store_true")
    a=ap.parse_args()
    report=collect(a.quote_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="quote_audits"},ensure_ascii=False))
    if a.strict and any(x["status"] not in (
        "PASS_PUBLIC_QUOTE_GIT_LOCK_BEFORE_SCHEDULED_FIRST_PITCH",
        "NO_EXTERNAL_GIT_ADD_COMMIT") for x in report["quote_audits"]):
        raise SystemExit("REJECTED_QUOTE_EVIDENCE")


if __name__=="__main__":main()

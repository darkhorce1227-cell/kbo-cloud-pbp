#!/usr/bin/env python3
"""Score externally precommitted 2026-10-18 PS Expected IP shadow, never refit.

Fails closed if prediction commit did not predate first pitch or if official
postgame audit is unavailable. Does not grade 10/14 or 10/17 retrospective rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from kbo_pbp import naver
from collect_rs_starter_lines import parse_ip

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research/postseason_expected_ip/2026_shadow/pregame_lock_2026_10_18_LG_HT.json"
AUDIT=ROOT/"state/postgame_score_audit_2026-10-18.json"
OUTPUT=ROOT/"research/postseason_expected_ip/2026_shadow/prospective_score_2026_10_18.json"


def _first_commit_datetime(lock: Path) -> datetime:
    rel=str(lock.resolve().relative_to(ROOT))
    proc=subprocess.run(
        ["git","log","--reverse","--format=%cI","--",rel],
        cwd=ROOT,text=True,capture_output=True,check=True
    )
    dates=[z.strip() for z in proc.stdout.splitlines() if z.strip()]
    if not dates:
        raise ValueError("No external Git commit contains the pregame snapshot")
    return datetime.fromisoformat(dates[0])


def score(lock_path: Path=LOCK, official_audit: Path=AUDIT, output: Path=OUTPUT) -> dict:
    if not lock_path.exists():
        raise FileNotFoundError("NO_PRECOMMITTED_PREGAME_SNAPSHOT")
    snap=json.loads(lock_path.read_text(encoding="utf-8"))
    first_pitch=datetime.fromisoformat(snap["first_pitch_kst"])
    creation=datetime.fromisoformat(snap["snapshot_created_at_kst"])
    commit=_first_commit_datetime(lock_path)
    if creation>=first_pitch or commit>=first_pitch:
        raise ValueError("PREGAME_LOCK_NOT_COMMITTED_BEFORE_FIRST_PITCH")
    if snap["game_date"]!="2026-10-18" or snap["game"]!="LG@HT":
        raise ValueError("Unexpected game in pregame snapshot")
    if snap["source_type"]!="FORECAST_BEFORE_GAME_OUTCOME":
        raise ValueError("Snapshot was not originally designated as prospective")
    if len(snap["forecast_rows"])!=2:
        raise ValueError("Expected two locked starter rows")
    official=json.loads(official_audit.read_text(encoding="utf-8"))
    if official.get("validation")!="PASS" or official.get("target_date")!="2026-10-18":
        raise ValueError("OFFICIAL_GAME_AUDIT_NOT_PASS")
    games=[g for g in official.get("games",[])
           if str(g.get("away_team"))=="LG" and str(g.get("home_team"))=="HT" and g.get("status")=="PASS"]
    if len(games)!=1:
        raise ValueError("Expected exactly one audited completed LG@HT game")
    game_pk=str(games[0]["game_pk"])
    relay=naver.fetch_relay(game_pk,1)
    scored=[]
    for locked in snap["forecast_rows"]:
        side=locked["side"]
        pitchers=(relay.get(f"{side}Lineup") or {}).get("pitcher") or []
        starters=[p for p in pitchers if int(p.get("seqno") or -1)==1]
        if len(starters)!=1:
            raise ValueError(f"{side}: starter not confirmed in completed relay")
        actual=starters[0]
        id_now=str(actual.get("pcode") or "")
        expected=str(locked["player_id"])
        if id_now!=expected:
            raise ValueError(f"STARTER_CHANGED: side={side} locked={expected} actual={id_now}; exclude from scoring")
        outs,ip=parse_ip(actual.get("inn"))
        if outs<0 or not math.isclose(ip*3,outs,abs_tol=1e-10):
            raise ValueError("Bad Naver completed innings")
        pred=locked["forecast"]
        if pred.get("status")=="UNAVAILABLE" or locked["rs_starter_avg_ip"] is None:
            scored.append({
                "team":locked["team"],"starter_id":expected,"starter_name":locked["starter_name"],
                "actual_ip":ip,"status":"UNAVAILABLE_BASELINE","included_in_metrics":False
            })
            continue
        mean=float(pred["mean_challenger"]["challenger_mean_ip"])
        baseline=float(locked["rs_starter_avg_ip"])
        events={}
        for event,rule in [("IP_GE_5",lambda x:x>=5),("IP_GE_6",lambda x:x>=6),("IP_LT_4",lambda x:x<4)]:
            prob=float(pred["event_probabilities"][event])
            outcome=int(rule(ip))
            events[event]={"forecast_probability":prob,"observed":outcome,"brier":(prob-outcome)**2}
        scored.append({
            "team":locked["team"],"starter_id":expected,"starter_name":locked["starter_name"],
            "actual_ip":ip,"actual_outs":outs,"pitch_count":actual.get("ballCount"),
            "baseline_rs_starter_avg_ip":baseline,
            "candidate_mean_ip":mean,
            "baseline_error":baseline-ip,"candidate_error":mean-ip,
            "baseline_abs_error":abs(baseline-ip),"candidate_abs_error":abs(mean-ip),
            "baseline_squared_error":(baseline-ip)**2,
            "candidate_squared_error":(mean-ip)**2,
            "events":events,
            "status":"PROSPECTIVE_SCORED","included_in_metrics":True
        })

    evaluated=[x for x in scored if x["included_in_metrics"]]
    if not evaluated:
        raise ValueError("No eligible prospective starts")
    n=len(evaluated)
    brier_by_event={
        event:sum(r["events"][event]["brier"] for r in evaluated)/n
        for event in ["IP_GE_5","IP_GE_6","IP_LT_4"]
    }
    report={
        "status":"PASS",
        "scope":"SINGLE_GAME_FIRST_PROSPECTIVE_SHADOW_ONLY",
        "first_pitch_kst":first_pitch.isoformat(),
        "snapshot_created_at_kst":creation.isoformat(),
        "first_snapshot_git_commit_at":commit.isoformat(),
        "snapshot_sha256":hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "game_pk":game_pk,"official_game_score_audit":"PASS",
        "completed_game_count":1,
        "prospective_starter_rows":len(scored),
        "eligible_starter_rows":n,
        "baseline_mae":sum(r["baseline_abs_error"] for r in evaluated)/n,
        "candidate_mae":sum(r["candidate_abs_error"] for r in evaluated)/n,
        "baseline_rmse":math.sqrt(sum(r["baseline_squared_error"] for r in evaluated)/n),
        "candidate_rmse":math.sqrt(sum(r["candidate_squared_error"] for r in evaluated)/n),
        "brier_by_event":brier_by_event,
        "rows":scored,
        "promotion_decision":"NO_MODEL_PROMOTION_FROM_ONE_GAME",
        "historical_refit":"FORBIDDEN",
        "betting_action":"NONE"
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lock",type=Path,default=LOCK)
    p.add_argument("--audit",type=Path,default=AUDIT)
    p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args()
    report=score(args.lock,args.audit,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False))


if __name__=="__main__":
    main()

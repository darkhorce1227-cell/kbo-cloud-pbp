#!/usr/bin/env python3
"""2026 KBO postseason Expected IP shadow: RS baseline, retrospective audit, pregame LOCK.

Strictly separate completed-games reconstructed after final from predictions committed
before first pitch. Source for RS is 2026 completed Naver RS starting lines (720 games,
1440 starts), never relief appearances. Stable player pcode joins are mandatory.

The first 2026 LIVE pregame lock target is 2026-10-18 LG@HT (14:00 KST)
with starters LG 임찬규, HT 올러. If the timestamp is late, NO snapshot is minted.
No market odds are consulted; no EV or automated bet.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from kbo_pbp import naver

from build_rs_starter_avg_ip import build_player_season, validate_appearances
from collect_rs_starter_lines import parse_ip
from forecast_ps_expected_ip_mean import load_canonical
from forecast_ps_expected_ip_tail import build_tail_forecast, load_verified_residuals

KST = ZoneInfo("Asia/Seoul")
GAME_IDS = {
    "2026-10-14": {"game_pk": "44441014OBLG02026", "away": "OB", "home": "LG", "round": "WC"},
    "2026-10-17": {"game_pk": "33331017LGHT02026", "away": "LG", "home": "HT", "round": "준PO"},
}
TARGET = {
    "date": "2026-10-18",
    "first_pitch_kst": "2026-10-18T14:00:00+09:00",
    "away": "LG",
    "home": "HT",
    "away_starter": "임찬규",
    "home_starter": "올러",
    "round": "준PO",
    "schedule_source": "2026-10-18 06:10 KST Newspim / 2026 KBO official postseason schedule",
}


def compute_live_row(
    rs_avg: pd.DataFrame,
    status: dict,
    contract: dict,
    residuals: list[float],
    calibration: dict,
    player_id: str,
    name: str,
    team: str,
    role: str = "ESTABLISHED_STARTER",
) -> dict:
    match = rs_avg.loc[rs_avg["starter_pcode"].astype(str) == str(player_id)]
    if len(match) > 1:
        raise ValueError(f"duplicate player baseline pcode {player_id}")
    if match.empty:
        forecast = build_tail_forecast(
            rs_ip=0.0, rs_starts=0, status=status, contract=contract,
            residuals=residuals, validation=calibration, role=role,
        )
        return dict(player_id=str(player_id), starter_name=name, team=team,
                    rs_starts=0, rs_starter_avg_ip=None, forecast=forecast,
                    baseline_match="UNAVAILABLE_NO_RS_STARTS")
    r = match.iloc[0]
    if not name or not team:
        raise ValueError("missing identity verification")
    forecast = build_tail_forecast(
        rs_ip=float(r["rs_starter_avg_ip"]), rs_starts=int(r["rs_starts"]),
        status=status, contract=contract, residuals=residuals,
        validation=calibration, role=role, round_name="준PO",
        manager_leash="unknown",
    )
    return dict(
        player_id=str(player_id), starter_name=name, team=team,
        rs_starts=int(r["rs_starts"]),
        rs_total_outs=int(r["rs_total_outs"]),
        rs_starter_avg_ip=float(r["rs_starter_avg_ip"]),
        baseline_match="MATCHED",
        forecast=forecast,
    )


def verified_name_to_id(rs: pd.DataFrame, team: str, name: str) -> str:
    x = rs.loc[
        (rs["team_code"].astype(str) == team)
        & (rs["starter_name"].astype(str) == name)
    ]
    ids = sorted(set(x["starter_pcode"].astype(str)))
    if len(ids) != 1:
        raise ValueError(f"2026 RS source cannot uniquely identify {team} {name}: {ids}")
    return ids[0]


def fetch_completed_starters(game_pk: str, away: str, home: str) -> list[dict]:
    relay = naver.fetch_relay(game_pk, 1)
    rows = []
    for side, team in (("away", away), ("home", home)):
        pitching = (relay.get(f"{side}Lineup") or {}).get("pitcher") or []
        first = [p for p in pitching if int(p.get("seqno") or -1) == 1]
        if len(first) != 1:
            raise ValueError(f"Game {game_pk} side {side}: expected one seqno=1, got {len(first)}")
        p = first[0]
        outs, ip = parse_ip(p.get("inn"))
        if outs < 0 or (ip*3-outs) != 0:
            raise ValueError("Invalid official starter IP/outs")
        rows.append({
            "game_pk":game_pk, "team":team, "side":side,
            "starter_id":str(p.get("pcode") or ""),
            "starter_name":str(p.get("name") or ""),
            "actual_outs":outs,"actual_ip":ip,"inn_raw":str(p.get("inn")),
            "pitch_count":p.get("ballCount"),"source":"Naver inning-1 completed pitching line",
        })
    if not all(x["starter_id"] and x["starter_name"] for x in rows):
        raise ValueError("Incomplete PS starter identity")
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rs-source",type=Path,default=Path("research/postseason_expected_ip/2026_shadow/source/2026_rs_starters.csv"))
    ap.add_argument("--rs-summary",type=Path,default=Path("research/postseason_expected_ip/2026_shadow/source/2026_rs_summary.json"))
    ap.add_argument("--output-dir",type=Path,default=Path("research/postseason_expected_ip/2026_shadow"))
    ap.add_argument("--skip-ps-fetch",action="store_true")
    args=ap.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rs=pd.read_csv(args.rs_source,dtype={"starter_pcode":str,"game_id":str,"team_code":str})
    summary=json.loads(args.rs_summary.read_text(encoding="utf-8"))
    quality=validate_appearances(rs,[2026])
    checks={
        "rs_upstream_status_pass":summary.get("status")=="PASS",
        "rs_upstream_720_games":summary.get("rs_games_collected")==720,
        "rs_upstream_1440_starts":summary.get("rs_starter_rows")==1440,
        "rs_validation_pass":quality.get("status")=="PASS",
        "rs_720_distinct_games":rs["game_id"].nunique()==720,
        "rs_1440_rows":len(rs)==1440,
        "rs_dates_not_after_2026_10_17":rs["game_date"].astype(str).max() < "2026-10-18",
        "rs_no_double_game_sides":not rs.duplicated(["game_id","side"]).any(),
    }
    if not all(checks.values()):
        raise ValueError(f"RS upstream provenance failed: {checks}")
    rs_avg=build_player_season(rs)
    rs_avg.to_csv(out/"rs_starter_avg_ip_2026.csv",index=False)
    status,contract=load_canonical()
    residuals,calibration=load_verified_residuals()

    retrospective=[]
    if not args.skip_ps_fetch:
        for date,game in GAME_IDS.items():
            for starter in fetch_completed_starters(game["game_pk"],game["away"],game["home"]):
                data=compute_live_row(rs_avg,status,contract,residuals,calibration,
                                      starter["starter_id"],starter["starter_name"],starter["team"])
                mean=data["forecast"]["mean_challenger"]["challenger_mean_ip"]
                retrospect={
                    "game_date":date, "round":game["round"],**starter,
                    "rs_starts":data["rs_starts"],
                    "rs_starter_avg_ip":data["rs_starter_avg_ip"],
                    "historical_mean_candidate_ip":mean,
                    "event_probabilities":data["forecast"]["event_probabilities"],
                    "error_candidate_minus_actual":None if mean is None else mean-starter["actual_ip"],
                    "source_type":"POSTGAME_RECONSTRUCTION_NOT_PROSPECTIVE",
                    "count_in_prospective_performance":False,
                }
                retrospective.append(retrospect)
        expected = {(g["game_pk"],side) for g in GAME_IDS.values() for side in ["away","home"]}
        actual = {(r["game_pk"],r["side"]) for r in retrospective}
        if actual != expected:
            raise ValueError("PS verified completed starter coverage mismatch")
        with (out/"retro_2026_ps_starters_through_1017.csv").open("w",encoding="utf-8",newline="") as f:
            fields=list(retrospective[0].keys())
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for row in retrospective:
                writer.writerow({
                    **row,"event_probabilities":json.dumps(row["event_probabilities"],ensure_ascii=False,sort_keys=True)
                })
        (out/"retro_2026_ps_starters_through_1017.json").write_text(
            json.dumps(retrospective,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
        )

    # A genuine prospective target must be timestamped AND externally committed
    # before first pitch. Never rewrite an existing snapshot.
    now=datetime.now(KST)
    kickoff=datetime.fromisoformat(TARGET["first_pitch_kst"])
    snapshot_path=out/"pregame_lock_2026_10_18_LG_HT.json"
    state="EXISTS_UNCHANGED" if snapshot_path.exists() else "NO_LOCK_LATE"
    if not snapshot_path.exists() and now < kickoff:
        pregame=[]
        for side,team,name in [("away",TARGET["away"],TARGET["away_starter"]),("home",TARGET["home"],TARGET["home_starter"])]:
            pcode=verified_name_to_id(rs,team,name)
            row=compute_live_row(rs_avg,status,contract,residuals,calibration,
                                 pcode,name,team)
            row.pop("forecast",None) if False else None
            pregame.append({"side":side,**row})
        payload={
            "status":"PREGAME_GENERATED_PENDING_GITHUB_COMMIT_BEFORE_KICKOFF",
            "snapshot_created_at_kst":now.isoformat(),
            "game_date":TARGET["date"],"first_pitch_kst":TARGET["first_pitch_kst"],
            "game":f"{TARGET['away']}@{TARGET['home']}",
            "round":TARGET["round"],
            "forecast_eligibility":"PRE_GAME_ONLY_IF_GITHUB_COMMIT_BEFORE_FIRST_PITCH",
            "source_type":"FORECAST_BEFORE_GAME_OUTCOME",
            "training_postseasons":"2016-2025",
            "ps_residual_n":len(residuals),
            "rs_2026_completed_starter_appearances":len(rs),
            "model_shift_ip":status["canonical_2026_shift_ip"],
            "forecast_rows":pregame,
            "prediction_use":"SHADOW_ONLY_NO_BET",
            "lineup_or_injury_uncertainty":"HUMAN_REVIEW_NOT_RESOLVED_BY_HISTORICAL_IP",
        }
        snapshot_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        state="CREATED_NEEDS_COMMIT_TIME_GATE"
    report={
        "status":"PASS","run_at_kst":now.isoformat(),
        "checks":checks,"rs_games":int(rs["game_id"].nunique()),"rs_starter_rows":len(rs),
        "rs_player_season_rows":len(rs_avg),
        "completed_ps_games_reconstructed":len(GAME_IDS) if not args.skip_ps_fetch else 0,
        "completed_ps_starter_rows_reconstructed":len(retrospective),
        "completed_ps_prospective_scored":0,
        "snapshot_today":state,
        "prospective_scores":"UNAVAILABLE_UNTIL_PRECOMMITTED_FORECAST_GAME_COMPLETED",
        "no_betting_actions":True,
        "canonical_historical_training_fixed_through_2025":True,
    }
    (out/"2026_shadow_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

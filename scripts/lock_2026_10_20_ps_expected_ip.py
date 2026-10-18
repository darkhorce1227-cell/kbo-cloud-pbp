#!/usr/bin/env python3
"""Immutable 2026-10-20 KIA@LG postseason Expected IP PREGAME shadow.

Read 2026 regular-season starts already collected (720 games / 1440 starts)
and the FROZEN 2016-25 postseason residual calibration. Never refit or
read odds. Refuse generation after preannounced 10/20 18:30 KST first pitch.

Research challenger only, not lineup-conditioned operational expected IP.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from build_2026_ps_eip_shadow import compute_live_row, verified_name_to_id
from forecast_ps_expected_ip_mean import load_canonical
from forecast_ps_expected_ip_tail import load_verified_residuals

ROOT=Path(__file__).resolve().parents[1]
FIRST_PITCH=datetime.fromisoformat("2026-10-20T18:30:00+09:00")
OUT=ROOT/"research/postseason_expected_ip/2026_shadow/pregame_lock_2026_10_20_HT_LG.json"
RS=ROOT/"research/postseason_expected_ip/2026_shadow/source/2026_rs_starters.csv"
BASE=ROOT/"research/postseason_expected_ip/2026_shadow/rs_starter_avg_ip_2026.csv"
STARTERS=[
    {"side":"away","team":"HT","name":"시라카와","expected_pcode":"54843",
     "rs_last_start":"2026-10-10","ps_prior_start":None,
     "specific_context":"FIRST_2026_PS_START; 10/20 manager confirmed; innings leash/pitch cap unverified"},
    {"side":"home","team":"LG","name":"카라스코","expected_pcode":"56103",
     "rs_last_start":"2026-10-09","ps_prior_start":"2026-10-14",
     "specific_context":"WC 10/14 6IP 1ER; 6 calendar days after last PS start; next game is elimination"},
]


def build(now=None,raw=RS,baseline=BASE):
    moment=now or datetime.now(ZoneInfo("Asia/Seoul"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("no external clock timezone")
    if moment>=FIRST_PITCH:
        raise ValueError("NO_NEW_PREGAME_SNAPSHOT_AFTER_FIRST_PITCH")
    rs=pd.read_csv(raw,dtype={"starter_pcode":str,"game_id":str,"team_code":str})
    av=pd.read_csv(baseline,dtype={"starter_pcode":str})
    if len(rs)!=1440 or rs["game_id"].nunique()!=720:
        raise ValueError("2026 RS 720/1440 coverage not valid")
    if rs["game_date"].astype(str).max()>="2026-10-18":
        raise ValueError("POSTSEASON_ROWS_IN_RS_STARTER_BASELINE")
    if len(av)<100 or av["starter_pcode"].duplicated().any():
        raise ValueError("invalid RS averages")
    status,contract=load_canonical()
    residuals,calibration=load_verified_residuals()
    if len(residuals)!=284:raise ValueError("unexpected frozen calibration length")
    rows=[]
    for item in STARTERS:
        pcode=verified_name_to_id(rs,item["team"],item["name"])
        if pcode!=item["expected_pcode"]:
            raise ValueError("starter identity or team was changed")
        x=compute_live_row(av,status,contract,residuals,calibration,pcode,
                           item["name"],item["team"])
        if x["rs_starts"]<10 or x["forecast"]["status"]=="UNAVAILABLE":
            raise ValueError("baseline / tail failed")
        rows.append({"side":item["side"],**x,
             "ps_prior_start":item["ps_prior_start"],
             "rs_last_start":item["rs_last_start"],
             "specific_context":item["specific_context"],
             "manager_confirmation":"2026-10-18 news announcements; official KBO 10/20 3rd game schedule"})
    return {
      "status":"PREGAME_GENERATED_PENDING_EXTERNAL_GIT_COMMIT_TIME_VERIFICATION",
      "game":"HT@LG","game_pk_planned":"33331020HTLG02026",
      "game_date":"2026-10-20","round":"준PO","scheduled_first_pitch_kst":FIRST_PITCH.isoformat(),
      "snapshot_created_at_kst":moment.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
      "source_type":"FORECAST_BEFORE_GAME_OUTCOME",
      "training_postseasons":"2016-2025","ps_residual_n":len(residuals),
      "source_rs_games":720,"source_rs_starts":1440,
      "model_shift_ip":status["canonical_2026_shift_ip"],
      "forecast_rows":rows,
      "model":"PS_EXPECTED_IP_EMPIRICAL_TAIL_2026_V1",
      "model_status":"CHALLENGER_ONLY_NOT_CHAMPION",
      "manager_leash_and_pitch_cap":"UNVERIFIED",
      "lineups":"NOT_CONFIRMED",
      "park":"JAMSIL",
      "market_odds_inspected":False,
      "market_source":"NONE",
      "initial_lock_immutability":"WRITE_X_FAIL_ON_EXISTING_FILE; external first Git commit must be before first pitch",
      "eligibility":"PROSPECTIVE_IF_ORIGINAL_GIT_COMMIT_BEFORE_FIRST_PITCH_AND_IDENTITY_MATCHED",
      "postgame_refit":"FORBIDDEN",
      "betting_action":"NONE",
      "starter_sources":[
        "https://www.newspim.com/news/view/20261018000126",
        "https://www.newspim.com/news/view/20261018000121",
        "https://www.koreabaseball.com/Schedule/Schedule.aspx?date=202610"
      ],
    }


def main():
    if OUT.exists():raise SystemExit("NO_OVERWRITE_EXISTING_2026_10_20_PS_LOCK")
    payload=build()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(json.dumps({"status":payload["status"],
      "created_at":payload["snapshot_created_at_kst"],
      "starter_rows":[{"name":r["starter_name"],
         "pcode":r["player_id"],
         "RS_avg_ip":r["rs_starter_avg_ip"],
         "IP_challenger":r["forecast"]["mean_challenger"]["challenger_mean_ip"],
         "tail":r["forecast"]["event_probabilities"]} for r in payload["forecast_rows"]]},ensure_ascii=False))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Reconstruct *three-way* 2026 KBO Proto WIN/1-run/LOSE from BETLIVE archives.

This source has KBO columns 승/무/패 where the middle means 1 RUN OR LESS,
not 'tie'; the official SportsToto 2023 WIN1LOSE announcement confirms that
0 or 1 run difference settles in the middle category.

CRITICAL: This is NOT the two-way general moneyline book, and NOT a
historically timestamp-verified purchase quote. Only for reconstructed
market-implied probability benchmarking after independent PBP result audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import requests

BASE="https://betlive.kr"
AGENT="KBOResearchAudit/1.0 (public archive; research only)"
TEAM_MAP={
    "두산":"OB","삼성":"SS","SSG":"SK","LG":"LG","KIA":"HT",
    "KT":"KT","NC":"NC","한화":"HH","롯데":"LT","키움":"WO",
}
FIELDS=[
    "game_pk","game_date","kickoff_hhmm_kst","home_team","away_team",
    "home_name","away_name","market_type","win_2plus_odds","within_1_odds",
    "lose_2plus_odds","inferred_outcome_from_pbp","proto_round",
    "source_round_url","source_observed_utc","source_price_temporality",
    "source_middle_header","settlement_source"
]
MARKET="FULL_WIN1LOSS_3WAY"
EXPLANATION_URL="https://www.sportstoto.co.kr/view.php?idx=2402&type=notice"


def parse_round(html: str, number:int, game_ids:set[str], observed:str):
    soup=BeautifulSoup(html,"html.parser")
    output=[];issues=[];n_kbo=0
    title=soup.title.get_text(" ",strip=True) if soup.title else ""
    if "프로토" not in title or str(number) not in title:
        raise ValueError(f"Wrong archived round title: {title[:110]}")
    headers=[[x.get_text(" ",strip=True) for x in tab.select("th")][:8] for tab in soup.select("table")]
    if not any(all(token in head for token in ["승","무","패"]) for head in headers):
        raise ValueError(f"Archived round columns not recognized: {headers}")
    for tr in soup.select("tr"):
        cells=tr.find_all("td",recursive=False)
        if len(cells)<6 or cells[1].get_text(" ",strip=True)!="KBO":
            continue
        n_kbo+=1
        try:
            d=cells[0].get_text(" ",strip=True)
            date_match=re.fullmatch(r"(\d{2})/(\d{2})\s+(\d{2}:\d{2})",d)
            if not date_match:
                raise ValueError("Cannot parse kickoff date and clock")
            mo,day,clock=date_match.groups()
            yyyy_mm_dd=f"2026-{mo}-{day}"
            names=cells[2].get_text(" ",strip=True)
            if " vs " not in names:
                raise ValueError(f"Unrecognized team delimiter: {names[:80]}")
            home_name,away_name=[s.strip() for s in names.split(" vs ",1)]
            home=TEAM_MAP.get(home_name);away=TEAM_MAP.get(away_name)
            if not home or not away or home==away:
                raise ValueError(f"Unrecognized KBO teams: {names}")
            gid_prefix=yyyy_mm_dd.replace("-","")+away+home
            matches=[g for g in game_ids if g.startswith(gid_prefix) and g.endswith("2026")]
            if len(matches)!=1:
                raise ValueError(f"Game PK not unique in validated KBO schedule: {len(matches)}")
            vals=[]
            for cell in cells[3:6]:
                text=cell.get_text(" ",strip=True)
                if not re.fullmatch(r"\d+(?:\.\d+)?",text):
                    raise ValueError(f"Invalid/missing KBO WIN1LOSE quote: {text}")
                v=float(text)
                if not math.isfinite(v) or v<=1.0:
                    raise ValueError(f"Nonphysical KBO quote {v}")
                vals.append(v)
            output.append({
               "game_pk":matches[0],"game_date":yyyy_mm_dd,"kickoff_hhmm_kst":clock,
               "home_team":home,"away_team":away,"home_name":home_name,"away_name":away_name,
               "market_type":MARKET,
               "win_2plus_odds":vals[0],"within_1_odds":vals[1],"lose_2plus_odds":vals[2],
               "inferred_outcome_from_pbp":"",
               "proto_round":number,
               "source_round_url":f"{BASE}/proto/260{number:03d}",
               "source_observed_utc":observed,
               "source_price_temporality":"HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED",
               "source_middle_header":"무_AS_PUBLISHED__INTERPRETED_WIN1LOSE_WITHIN_1",
               "settlement_source":"INDEPENDENT_PBP_ONLY_PUBLISHED_PAGE_HAS_NO_RESULT",
            })
        except (ValueError,TypeError,IndexError) as err:
            issues.append({"round":number,"row":tr.get_text(" ",strip=True)[:130],
                           "reason":str(err)[:200]})
    return output,issues,n_kbo


def collect(first:int,last:int,output_dir:Path,pause=0.45)->dict:
    m=json.loads(Path("state/validated_game_pks_2026.json").read_text(encoding="utf-8"))
    if m.get("year")!=2026 or m.get("game_count",0)<700:
        raise ValueError("Cannot trust source without a validated 2026 game ID index")
    game_ids=set(str(s) for s in m["game_pks"])
    sess=requests.Session()
    sess.headers["User-Agent"]=AGENT
    rob=sess.get(BASE+"/robots.txt",timeout=20)
    if rob.status_code!=200:
        raise ValueError(f"robots.txt unverified HTTP={rob.status_code}")
    parser=RobotFileParser()
    parser.parse(rob.content.decode("utf-8","replace").splitlines())
    observed=datetime.now(timezone.utc).isoformat()
    all_rows=[];all_issues=[];rounds=[]
    for number in range(first,last+1):
        url=f"{BASE}/proto/260{number:03d}"
        if not parser.can_fetch(AGENT,url):
            rounds.append({"round":number,"status":"ROBOTS_DISALLOW","kbo_rows":0})
            continue
        try:
            r=sess.get(url,timeout=22)
            if r.status_code==404:
                rounds.append({"round":number,"status":"NOT_FOUND_404","kbo_rows":0})
                continue
            r.raise_for_status()
            rows,issues,n_kbo=parse_round(r.content.decode("utf-8","replace"),
                                         number,game_ids,observed)
            all_rows.extend(rows);all_issues.extend(issues)
            rounds.append({"round":number,"status":"PARSED","kbo_rows":n_kbo,
                           "strict_game_id_market_rows":len(rows),
                           "issues":len(issues)})
        except requests.RequestException as err:
            rounds.append({"round":number,"status":"HTTP_ERROR","error":str(err)[:130]})
        except ValueError as err:
            rounds.append({"round":number,"status":"SCHEMA_OR_AUDIT_ERROR","error":str(err)[:130]})
        print(json.dumps(rounds[-1],ensure_ascii=False),flush=True)
        time.sleep(pause)

    dedup={};duplicates=[];conflicts=[]
    for row in sorted(all_rows,key=lambda x:(x["proto_round"],x["game_pk"])):
        gid=row["game_pk"]
        if gid in dedup:
            duplicates.append({"game_pk":gid,"round_a":dedup[gid]["proto_round"],
                               "round_b":row["proto_round"]})
            if (row["win_2plus_odds"],row["within_1_odds"],row["lose_2plus_odds"])!=(
                dedup[gid]["win_2plus_odds"],dedup[gid]["within_1_odds"],
                dedup[gid]["lose_2plus_odds"]):
                conflicts.append({"game_pk":gid,"reason":"CROSS_ROUND_QUOTES_DIFFER_NO_SINGLE_PREGAME_TIMESTAMP",
                                  "round_a":dedup[gid]["proto_round"],
                                  "round_b":row["proto_round"]})
            continue
        dedup[gid]=row
    report={
        "status":"HISTORICAL_3WAY_QUOTES_NO_PREGAME_TIMESTAMP",
        "probed_at_utc":observed,"first_round":first,"last_round":last,
        "rounds":rounds,
        "rounds_parsed":sum(z["status"]=="PARSED" for z in rounds),
        "rounds_404":[z["round"] for z in rounds if z["status"]=="NOT_FOUND_404"],
        "rounds_failed":[z for z in rounds if z["status"] not in ("PARSED","NOT_FOUND_404")],
        "kbo_rows_in_pages":sum(z.get("kbo_rows",0) for z in rounds),
        "source_rows_strict_game_pk_join":len(all_rows),
        "distinct_game_markets":len(dedup),
        "cross_round_duplicates":len(duplicates),
        "cross_round_quote_disagreements":conflicts[:80],
        "issues_count":len(all_issues),"issues":all_issues[:110],
        "market_id":"FULL_WIN1LOSS_3WAY",
        "market_outcomes":{
            "HOME_2PLUS":"home run diff >= 2",
            "WITHIN_1":"absolute run diff <=1 including ties",
            "AWAY_2PLUS":"away run diff >= 2",
        },
        "three_way_middle_official_rule_url":EXPLANATION_URL,
        "must_independently_verify_scores_with_pbp":True,
        "historical_odds_executable_at_first_pitch_unverified":True,
        "never_join_as_regular_ml2":True,
        "not_genuine_prospective_bet_roi":True,
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    csv_file=output_dir/f"betlive_2026_r{first}_r{last}_win1loss3_archived.csv"
    audit_file=output_dir/f"betlive_2026_r{first}_r{last}_win1loss3_source_audit.json"
    with csv_file.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=FIELDS);wr.writeheader()
        wr.writerows(sorted(dedup.values(),key=lambda x:(x["game_date"],x["game_pk"])))
    audit_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ("rounds","issues","cross_round_quote_disagreements","rounds_failed")},
                      ensure_ascii=False),flush=True)
    if len(dedup)<30 or report["rounds_failed"]:
        raise SystemExit("SOURCE_COVERAGE_GATE_FAIL: preserve source audit, no calibration")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--first",type=int,default=40)
    ap.add_argument("--last",type=int,default=104)
    ap.add_argument("--pause",type=float,default=0.45)
    ap.add_argument("--output-dir",type=Path,default=Path("research/market_fair_odds"))
    x=ap.parse_args()
    if not (1<=x.first<=x.last<=123) or x.last-x.first>=70:
        raise ValueError("Cap each archive run to at most 70 consecutive rounds")
    collect(x.first,x.last,x.output_dir,x.pause)


if __name__=="__main__":
    main()

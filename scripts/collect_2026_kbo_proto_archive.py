#!/usr/bin/env python3
"""Reconstruct 2026 KBO Proto prices from public archived round HTML.

No fictional price_asof_kst or CLEAN PRE records. Every result is a
RETROSPECTIVE archived quote with unknown pre-first-pitch availability time.
Scores must join the repo's independent Naver game_pk manifest before use.

Page structure verified in October 2026: SharpPick 2026/116 table tr[data-sp]
contains team pairs, regular 2-way, handicap line, total line, final score.
All network requests honor robots.txt and use a modest delay.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import math
import re
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import requests

BASE="https://sharppick.app"
USER_AGENT="KBOResearchAudit/1.0 (research-only, archived odds, noncommercial)"
TEAM_NAMES={
 "두산 베어스":"OB","NC 다이노스":"NC","삼성 라이온즈":"SS",
 "한화 이글스":"HH","롯데 자이언츠":"LT","키움 히어로즈":"WO",
 "SSG 랜더스":"SK","LG 트윈스":"LG","KIA 타이거즈":"HT",
 "KT 위즈":"KT",
}
FIELDNAMES=[
 "game_pk","game_date","kickoff_hhmm_kst","home_team","away_team",
 "home_name","away_name","home_score","away_score",
 "market_type","line","odds_json","opening_odds_json","settlement_status",
 "settled_outcome","proto_round","source_match_url","source_round_url",
 "source_observed_utc","source_price_temporality","opening_value_provenance"
]
MARKET_OPTIONS={
 "FULL_ML2":("HOME","AWAY"),
 "FULL_HC2":("HOME","AWAY"),
 "FULL_OU":("UNDER","OVER")
}
LABELS={"승":"HOME","패":"AWAY","언더":"UNDER","오버":"OVER"}


def get_allowed(url, session, pause=0.8):
    rp=RobotFileParser()
    rob=session.get(BASE+"/robots.txt",timeout=18)
    if rob.status_code!=200:
        raise ValueError(f"Cannot confirm robots rules HTTP {rob.status_code}")
    rp.parse(rob.content.decode("utf-8","replace").splitlines())
    if not rp.can_fetch(USER_AGENT,url):
        raise ValueError("robots.txt disallows archived Proto round fetch")
    time.sleep(pause)
    response=session.get(url,timeout=25)
    response.raise_for_status()
    return response.content.decode("utf-8","replace")


def _read_odds(td):
    if td is None:
        raise ValueError("Missing market odds cell")
    # each selected span directly under td or its single wrapping div
    parent=td.find("div",recursive=False) or td
    parsed={}
    previous={}
    inferred_opening=False
    for s in parent.find_all("span",recursive=False):
        label_tag=s.select_one(".rd-lb")
        if not label_tag:continue
        label=LABELS.get(label_tag.get_text(" ",strip=True))
        if label is None:continue
        text=s.get_text(" ",strip=True)
        number=re.search(r"(?<![\d.])\d+(?:\.\d+)?",text)
        if not number:continue
        current=float(number.group())
        if not math.isfinite(current) or current<=1:raise ValueError("Invalid decimal odds")
        parsed[label]=current
        change=s.select_one(".dn,.up")
        if change is None:
            previous[label]=current
            inferred_opening=True
        else:
            change_num=re.search(r"\d+(?:\.\d+)?",change.get_text(" ",strip=True))
            if not change_num: raise ValueError("Bad opening price marker")
            previous[label]=float(change_num.group())
    return parsed,previous,inferred_opening


def _score_market(market_type, line, home, away):
    if market_type=="FULL_ML2":
        if home==away:return "PUSH",None
        return "SETTLED","HOME" if home>away else "AWAY"
    if market_type=="FULL_HC2":
        net=home+float(line)-away
        if math.isclose(net,0,abs_tol=1e-9):return "PUSH",None
        return "SETTLED","HOME" if net>0 else "AWAY"
    if market_type=="FULL_OU":
        net=(home+away)-float(line)
        if math.isclose(net,0,abs_tol=1e-9):return "PUSH",None
        return "SETTLED","OVER" if net>0 else "UNDER"
    raise ValueError("Unsupported market")


def _date_from_prior_header(tr,year):
    for earlier in tr.find_all_previous("tr",limit=180):
        th=earlier.find("th")
        if not th:continue
        text=th.get_text(" ",strip=True)
        m=re.search(r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})\s*\(",text)
        if m:
            month,day=map(int,m.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    raise ValueError("No round day header preceding KBO market row")


def extract_round(html,round_no,manifest,now):
    soup=BeautifulSoup(html,"html.parser")
    out=[]; issues=[]; n_kbo=0
    for tr in soup.select('tr[data-sp="baseball"]'):
        tds=tr.find_all("td",recursive=False)
        if len(tds)<7 or "KBO" not in tds[1].get_text(" ",strip=True):continue
        n_kbo+=1
        try:
            url=tds[1].select_one('a[href^="/match/"]')
            if url is None:raise ValueError("Missing source match URL")
            pair=url.get_text(" ",strip=True).split(" 대 ")
            if len(pair)!=2:raise ValueError("Unrecognized home/away team text")
            home_name,away_name=pair
            home=TEAM_NAMES.get(home_name);away=TEAM_NAMES.get(away_name)
            if not home or not away or home==away:raise ValueError(f"Unknown team names {pair}")
            day=_date_from_prior_header(tr,2026)
            clock=tds[0].get_text(" ",strip=True)
            if not re.fullmatch(r"\d{2}:\d{2}",clock):raise ValueError("No HH:MM kickoff")
            outcome_text=tds[6].get_text(" ",strip=True)
            score=re.search(r"\b(\d+)\s*:\s*(\d+)\b",outcome_text)
            if not score:
                issues.append({"round":round_no,"date":day,"pair":pair,"reason":"NOT_FINAL_OR_INVALID_SCORE"})
                continue
            home_score,away_score=map(int,score.groups())
            key=day.replace("-","")+away+home
            candidates=[gid for gid in manifest if gid.startswith(key) and gid.endswith("2026")]
            if len(candidates)!=1:
                issues.append({"round":round_no,"date":day,"pair":pair,
                               "reason":"GAME_PK_JOIN_NOT_EXACT",
                               "manifest_candidates":candidates[:4]})
                continue
            gid=candidates[0]
            general,gen_open,gen_infer=_read_odds(tds[2])
            handicaps=tds[4].select("div")
            totals=tds[5].select("div")
            offer_list=[("FULL_ML2","",general,gen_open,gen_infer)]
            for d in handicaps:
                line=d.find("b",recursive=False)
                if not line:continue
                odds,opening,inferred=_read_odds(d)
                offer_list.append(("FULL_HC2",line.get_text(" ",strip=True),
                                   odds,opening,inferred))
            for d in totals:
                line=d.find("b",recursive=False)
                if not line:continue
                odds,opening,inferred=_read_odds(d)
                offer_list.append(("FULL_OU",line.get_text(" ",strip=True),
                                   odds,opening,inferred))
            url_round=f"{BASE}/proto/2026/{round_no}/"
            for typ,line,odds,open_odds,inferred in offer_list:
                if set(odds)!=set(MARKET_OPTIONS[typ]) or set(open_odds)!=set(MARKET_OPTIONS[typ]):
                    raise ValueError(f"Incomplete {typ} price book: {odds}")
                for name in odds:
                    if not math.isfinite(odds[name]) or odds[name]<=1:raise ValueError("Invalid market price")
                st,win=_score_market(typ,line,home_score,away_score)
                out.append(dict(game_pk=gid,game_date=day,kickoff_hhmm_kst=clock,
                   home_team=home,away_team=away,home_name=home_name,away_name=away_name,
                   home_score=home_score,away_score=away_score,
                   market_type=typ,line=line,
                   odds_json=json.dumps(odds,ensure_ascii=False,sort_keys=True),
                   opening_odds_json=json.dumps(open_odds,ensure_ascii=False,sort_keys=True),
                   settlement_status=st,settled_outcome=win or "",
                   proto_round=round_no,source_match_url=BASE+url.get("href"),
                   source_round_url=url_round,source_observed_utc=now,
                   source_price_temporality="HISTORICAL_PAGE_RETROSPECTIVE_PREMATCH_PRICE_TIME_UNVERIFIED",
                   opening_value_provenance=("NO_CHANGE_ASSUMED_EQUAL_TO_DISPLAY" if inferred else
                                             "EXPLICIT_ARROW_MARKER_WHEN_PRESENT")))
        except (ValueError,TypeError,IndexError) as exc:
            issues.append({"round":round_no,"row":tr.get_text(" ",strip=True)[:200],
                           "reason":f"{type(exc).__name__}:{str(exc)[:150]}"})
    return out,issues,n_kbo


def run(first,last,dry_run=False):
    manifest_path=Path("state/validated_game_pks_2026.json")
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m["year"]==2026 and m["game_count"]>=720
    manifest=set(map(str,m["game_pks"]))
    session=requests.Session()
    session.headers.update({"User-Agent":USER_AGENT})
    now=datetime.now(timezone.utc).isoformat()
    records=[];issues=[];rounds=[]
    for round_no in range(first,last+1):
        url=f"{BASE}/proto/2026/{round_no}/"
        try:
            html=get_allowed(url,session)
            rows,problems,kbo=extract_round(html,round_no,manifest,now)
            records.extend(rows);issues.extend(problems)
            rounds.append({"round":round_no,"http":"OK","kbo_match_rows":kbo,
                          "market_rows":len(rows),"problems":len(problems)})
        except requests.exceptions.RequestException as ex:
            rounds.append({"round":round_no,"http":"FETCH_ERROR","error":str(ex)[:200]})
        print(json.dumps(rounds[-1],ensure_ascii=False),flush=True)
    key=lambda r:(r["game_pk"],r["market_type"],r["line"])
    dedup={}
    duplicate_round_observations=0; disagreement=[]
    # Earliest numbered round for a specific market is the primary descriptive quote.
    # One score per game-market, even if the book relists across rounds.
    for r in sorted(records,key=lambda x:(x["proto_round"],x["game_pk"])):
        k=key(r)
        if k in dedup:
            duplicate_round_observations+=1
            prev=dedup[k]
            if (r["home_score"],r["away_score"],r["settlement_status"])!=(
                prev["home_score"],prev["away_score"],prev["settlement_status"]):
                disagreement.append({"game_pk":r["game_pk"],"market_type":r["market_type"],
                                     "line":r["line"],"rounds":[prev["proto_round"],r["proto_round"]],
                                     "reason":"INCONSISTENT_FINAL_ACROSS_ROUNDS"})
            continue
        dedup[k]=r
    report={
       "status":"SOURCE_RECONSTRUCTED_UNVERIFIED_PRICE_TIMESTAMP",
       "source":BASE,"year":2026,"first_round":first,"last_round":last,
       "capture_utc":now,"rounds":rounds,
       "rows_including_multi_round":len(records),
       "deduplicated_markets":len(dedup),
       "unique_games":len(set(r["game_pk"] for r in dedup.values())),
       "kbo_match_rows_in_source":sum(x.get("kbo_match_rows",0) for x in rounds),
       "duplicate_round_market_observations":duplicate_round_observations,
       "issues":issues[:150],"issue_count":len(issues),
       "cross_round_disagreements":disagreement[:100],
       "duplicate_game_market_ids":len(dedup)!=len(set(key(x) for x in dedup.values())),
       "market_type_counts":{typ:sum(r["market_type"]==typ for r in dedup.values())
                             for typ in MARKET_OPTIONS},
       "completed_rounds":sum(x.get("http")=="OK" for x in rounds),
       "all_rounds_collected":len(rounds)==last-first+1 and
           all(x.get("http")=="OK" for x in rounds),
       "not_validated_as_original_pregame_quote":True,
       "may_be_used_for_clean_pre_performance":False,
       "source_warning":"Postgame historical odds pages may not prove exact pre-first-pitch availability or price time.",
    }
    if disagreement:report["status"]="SOURCE_SCORE_CONFLICT_REQUIRES_REVIEW"
    out=Path("research/market_fair_odds")
    out.mkdir(parents=True,exist_ok=True)
    csv_file=out/f"sharppick_2026_proto_r{first}_r{last}_kbo_reconstructed.csv"
    audit_file=out/f"sharppick_2026_proto_r{first}_r{last}_audit.json"
    with csv_file.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=FIELDNAMES)
        wr.writeheader()
        wr.writerows(sorted(dedup.values(),key=lambda a:(a["game_date"],a["game_pk"],a["market_type"],a["line"])))
    audit_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("issues","rounds","cross_round_disagreements")},
                     ensure_ascii=False))
    if report["status"]=="SOURCE_SCORE_CONFLICT_REQUIRES_REVIEW":
        raise SystemExit(2)
    if not dry_run and (not report["all_rounds_collected"] or report["deduplicated_markets"]<8):
        raise SystemExit("Audit incomplete / insufficient book; retained data for review")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--first",type=int,default=112)
    ap.add_argument("--last",type=int,default=122)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    if not (1<=args.first<=args.last<=123) or args.last-args.first+1>15:
        raise ValueError("Limit to 15 contiguous known rounds per run")
    run(args.first,args.last,args.dry_run)


if __name__=="__main__":
    main()

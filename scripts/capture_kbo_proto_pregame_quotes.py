#!/usr/bin/env python3
"""Capture immutable, independently timestamped PUBLIC KBO Proto price displays.

NOT official Betman executable quotes. NOT a market-blind CLEAN PRE model.
Captures only fixtures >=20 minutes before the source-scheduled first pitch,
keeps unknown future game_pk null, and never rewrites prior-price files.
Git first ADD commit must independently precede first pitch for prospective
public-quote provenance. Rescheduling and actual first pitch need later audit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup

BASE="https://sharppick.app"
INDEX=BASE+"/proto/"
AGENT="KBOIndependentMarketResearch/1.0 (public-display timestamp evidence only)"
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/"state/pregame_market_quotes"
TEAMS={"두산 베어스":"OB","NC 다이노스":"NC","삼성 라이온즈":"SS",
       "한화 이글스":"HH","롯데 자이언츠":"LT","키움 히어로즈":"WO",
       "SSG 랜더스":"SK","LG 트윈스":"LG","KIA 타이거즈":"HT","KT 위즈":"KT"}
FIELDS={"FULL_ML2":("HOME","AWAY"),"FULL_HC2":("HOME","AWAY"),
        "FULL_OU":("UNDER","OVER")}
LABEL={"승":"HOME","패":"AWAY","언더":"UNDER","오버":"OVER"}


def now_utc():
    return datetime.now(timezone.utc)


def _book(cell, allowed):
    if cell is None:
        raise ValueError("missing market cell")
    parent=cell.find("div",recursive=False) or cell
    odds={}
    for sp in parent.find_all("span",recursive=False):
        lb=sp.select_one(".rd-lb")
        if lb is None:continue
        side=LABEL.get(lb.get_text(" ",strip=True))
        if side not in allowed:continue
        cloned=BeautifulSoup(str(sp),"html.parser").find("span")
        for old in cloned.select(".dn,.up"):old.decompose()
        label=cloned.select_one(".rd-lb")
        if label:label.decompose()
        txt=cloned.get_text(" ",strip=True)
        if not re.fullmatch(r"\d+(?:\.\d+)?",txt):
            raise ValueError("unreadable displayed price")
        price=float(txt)
        if not 1<price<1000 or side in odds:
            raise ValueError("invalid or duplicate price")
        odds[side]=price
    if set(odds)!=set(allowed):
        raise ValueError("incomplete market price book")
    return odds


def _date_header(tr):
    for prev in tr.find_all_previous("tr",limit=220):
        for th in prev.find_all("th"):
            m=re.search(r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})\s*\(",th.get_text(" ",strip=True))
            if m:
                month,day=map(int,m.groups())
                return f"2026-{month:02d}-{day:02d}"
    raise ValueError("no source day header")


def parse_quotes(html,page_url,observed,min_buffer_minutes=20,max_days_ahead=9):
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("observed must be timezone aware")
    observed=observed.astimezone(timezone.utc)
    soup=BeautifulSoup(html,"html.parser")
    title=soup.title.get_text(" ",strip=True) if soup.title else ""
    if "프로토" not in title or "회차" not in title:
        raise ValueError("not a Proto round page")
    found,issues=[],[]
    for tr in soup.select('tr[data-sp="baseball"]'):
        td=tr.find_all("td",recursive=False)
        if len(td)<7 or "KBO" not in td[1].get_text(" ",strip=True):continue
        try:
            anchor=td[1].select_one('a[href^="/match/"]')
            if anchor is None:raise ValueError("missing source match URL")
            names=anchor.get_text(" ",strip=True).split(" 대 ")
            if len(names)!=2:raise ValueError("bad KBO team pair")
            home,away=[TEAMS.get(x) for x in names]
            if not home or not away or home==away:raise ValueError("unknown team")
            day=_date_header(tr)
            clock=td[0].get_text(" ",strip=True)
            if not re.fullmatch(r"\d{2}:\d{2}",clock):
                raise ValueError("bad source kickoff")
            scheduled=datetime.fromisoformat(f"{day}T{clock}:00+09:00")
            if observed+timedelta(minutes=min_buffer_minutes)>=scheduled.astimezone(timezone.utc):
                continue
            if scheduled>observed+timedelta(days=max_days_ahead):continue
            if re.search(r"\b\d+\s*:\s*\d+\b",td[6].get_text(" ",strip=True)):
                raise ValueError("future source listing already displays final score")
            offers=[("FULL_ML2","",td[2])]
            for kind,cell in [("FULL_HC2",td[4]),("FULL_OU",td[5])]:
                for div in cell.find_all("div",recursive=False):
                    tag=div.find("b",recursive=False)
                    if tag is None:continue
                    line=tag.get_text(" ",strip=True)
                    if not -50<=float(line)<=50:raise ValueError("bad numeric line")
                    offers.append((kind,line,div))
            collected=[]
            for kind,line,cell in offers:
                book=_book(cell,FIELDS[kind])
                collected.append({
                    "capture_type":"PUBLIC_DISPLAYED_PRICE_OBSERVED_PRE_SCHEDULED_START",
                    "market_id":f"{day}|{away}>{home}|{kind}|{line}",
                    "event_date_kst":day,"home_team":home,"away_team":away,
                    "home_name":names[0],"away_name":names[1],
                    "scheduled_first_pitch_kst":scheduled.isoformat(),
                    "market_type":kind,"market_line":line,"decimal_odds":book,
                    "outcomes":list(FIELDS[kind]),"source_round_url":page_url,
                    "source_match_url":urljoin(page_url,anchor.get("href")),
                    "source_row_status":td[6].get_text(" ",strip=True),
                    "first_observed_at_utc":observed.isoformat(),
                    "displayed_price_is_executable_at_betman":False,
                    "time_verified":"OBSERVATION_TIME_ONLY_NOT_BOOK_OPENING_TIME",
                    "future_game_pk":None,"is_clean_pre_model_prediction":False,
                    "roi_backtest_eligibility":"WAIT_OFFICIAL_BETTING_AVAILABILITY_AND_ACTUAL_START",
                })
            found.extend(collected)
        except (ValueError,IndexError,TypeError) as err:
            issues.append({"row":tr.get_text(" ",strip=True)[:150],
                           "error":str(err)[:170]})
    return found,issues


def latest_round_links(index_html,count=3):
    soup=BeautifulSoup(index_html,"html.parser")
    rounds=set()
    for a in soup.select("a[href]"):
        m=re.search(r"/proto/2026/(\d{1,3})/?(?:[?#].*)?$",a.get("href",""))
        if m and 1<=int(m.group(1))<=150:rounds.add(int(m.group(1)))
    return [f"{BASE}/proto/2026/{n}/" for n in sorted(rounds,reverse=True)[:count]]


def source_get(client,url,robots):
    if not robots.can_fetch(AGENT,url):raise ValueError("robots disallow market page")
    response=client.get(url,timeout=24,headers={"User-Agent":AGENT})
    response.raise_for_status()
    captured=now_utc() # immediately after complete HTTP response
    if "text/html" not in response.headers.get("Content-Type","text/html"):
        raise ValueError("non-HTML quote source")
    return response.content.decode("utf-8","replace"),captured,response.headers.get("Date","")


def _stable_fingerprint(item):
    keys=("market_id","scheduled_first_pitch_kst","decimal_odds",
          "source_round_url","source_match_url")
    content={k:item[k] for k in keys}
    return hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,
                                    separators=(",",":")).encode()).hexdigest()


def capture(out_dir=DEFAULT_OUT,dry_run=False,source_rounds=3,session=None):
    client=session or requests.Session()
    client.headers["User-Agent"]=AGENT
    res=client.get(BASE+"/robots.txt",timeout=17)
    if res.status_code!=200:raise ValueError("cannot verify robots.txt")
    rp=RobotFileParser();rp.parse(res.content.decode("utf-8","replace").splitlines())
    index,at,_=source_get(client,INDEX,rp)
    urls=latest_round_links(index,source_rounds)
    if not urls:raise ValueError("no advertised 2026 Proto round links")
    quotes,issues,pages=[],[],[]
    for url in urls:
        html,seen,http_date=source_get(client,url,rp)
        extracted,errors=parse_quotes(html,url,seen)
        sha=hashlib.sha256(html.encode("utf-8")).hexdigest()
        for quote in extracted:
            quote["source_round_response_sha256"]=sha
            quote["source_server_date_header"]=http_date
        quotes.extend(extracted)
        issues.extend({"source":url,**e} for e in errors)
        pages.append({"source":url,"observed_utc":seen.isoformat(),
                      "html_sha256":sha,"eligible_markets":len(extracted),
                      "parse_errors":len(errors)})
    distinct={}
    for q in quotes:
        if q["market_id"] not in distinct:distinct[q["market_id"]]=q
    created=[];existing=0
    for quote in distinct.values():
        fp=_stable_fingerprint(quote)
        kind=quote["market_type"]
        line=re.sub(r"[^A-Za-z0-9]+","-",quote["market_line"] or "NA")
        folder=f"{quote['event_date_kst']}_{quote['away_team']}_{quote['home_team']}"
        path=out_dir/folder/f"{kind}_{line}_{fp[:20]}.json"
        if path.exists():
            existing+=1;continue
        if not dry_run:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8") as handle:
                json.dump({**quote,"snapshot_fingerprint_sha256":fp,
                   "original_prospective_label":"PUBLIC_PRICE_CAPTURE_REQUIRES_FIRST_COMMIT_TIME_GATE"},
                   handle,ensure_ascii=False,indent=2)
                handle.write("\n")
        created.append({"path":str(path),"market_id":quote["market_id"],
            "scheduled_first_pitch_kst":quote["scheduled_first_pitch_kst"]})
    return {"status":"PUBLIC_MARKET_CAPTURE_NOT_BETTABLE_PRICE_CERTIFICATION",
        "source_index":INDEX,"index_observed_utc":at.isoformat(),
        "round_urls":urls,"round_pages":pages,"parse_issues":issues[:80],
        "markets_seen_before_start":len(distinct),"new_quote_files":created,
        "already_identical_quote_files":existing,
        "executability":"NOT_VERIFIED",
        "model_clean_pre":"MUST_BE_SEPARATELY_PRECOMMITTED_BEFORE_MARKET_INSPECTION",
        "actual_first_pitch":"NOT_VERIFIED_FOR_FUTURE_FIXTURES"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--source-rounds",type=int,default=3)
    a=p.parse_args()
    if not 1<=a.source_rounds<=5:raise ValueError("max five latest rounds")
    print(json.dumps(capture(a.out_dir,a.dry_run,a.source_rounds),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

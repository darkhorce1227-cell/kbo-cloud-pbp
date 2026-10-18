#!/usr/bin/env python3
"""Small public-site discovery; no login, bypass, or odds data claims.

Never store page HTML or fetch more than a few public pages. Honor robots.txt.
This action is only to determine which historical quote representations exist.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

TARGETS=[
    "https://spogod.com/round/proto/",
    "https://spogod.com/today/?d=2026-10-01",
    "https://sharppick.app/proto/",
    "https://todaylotto.kr/toto/proto",
]
HEADERS={"User-Agent":"KBOResearchAudit/1.0 (public-source discovery; no automated betting)"}


def probe_one(url:str):
    r=requests.Session()
    u=urlparse(url)
    robots=urljoin(f"{u.scheme}://{u.netloc}", "/robots.txt")
    item={"url":url,"robots_url":robots,"attempt_at_utc":datetime.now(timezone.utc).isoformat()}
    try:
        rr=r.get(robots,headers=HEADERS,timeout=15)
        item["robots_http_status"]=rr.status_code
        if rr.status_code!=200:
            item["status"]="NO_ROBOTS_PERMISSION_CONFIRMED"
            return item
        rp=RobotFileParser()
        rp.parse(rr.text.splitlines())
        item["robots_allow"]=rp.can_fetch(HEADERS["User-Agent"],url)
        if not item["robots_allow"]:
            item["status"]="ROBOTS_DISALLOW"
            return item
        response=r.get(url,headers=HEADERS,timeout=22)
        item["http_status"]=response.status_code
        item["content_type"]=response.headers.get("content-type")
        item["html_bytes"]=len(response.content)
        item["status"]="HTTP_ERROR" if response.status_code!=200 else "ACCESSIBLE"
        if response.status_code!=200:
            return item
        soup=BeautifulSoup(response.text,"html.parser")
        item["title"]=soup.title.get_text(" ",strip=True) if soup.title else None
        headings=[x.get_text(" ",strip=True) for x in soup.select("h1,h2,h3")]
        item["headings"]=headings[:35]
        linkset=[]
        for a in soup.select("a[href]"):
            text=a.get_text(" ",strip=True)
            href=urljoin(response.url,a.get("href"))
            if any(t in (text+" "+href).lower() for t in ["proto","프로토","회차","kbo","야구","/match/"]):
                linkset.append({"text":text[:90],"url":href[:220]})
        item["relevant_links"]=linkset[:65]
        blocks=[]
        for el in soup.find_all(["div","tr","article","li","a"]):
            t=el.get_text(" ",strip=True)
            if "KBO" in t and 15<=len(t)<=480 and len(list(el.children))<=15:
                blocks.append(t[:440])
            if len(blocks)>=12:
                break
        item["kbo_html_text_examples"]=blocks
        item["tables"]=[{"heads":[x.get_text(" ",strip=True) for x in tab.select("th")][:10],
                         "first_rows":[row.get_text(" ",strip=True)[:300] for row in tab.select("tr")[:3]]}
                        for tab in soup.select("table")[:7]]
        item["jsonld_count"]=len(soup.select('script[type="application/ld+json"]'))
        item["next_data_present"]=bool(soup.select("script#__NEXT_DATA__"))
        item["script_paths"]=[urljoin(url,s.get("src")) for s in soup.select("script[src]")][:20]
    except Exception as e:
        item["status"]="FETCH_ERROR"
        item["error"]=f"{type(e).__name__}: {str(e)[:160]}"
    return item


def main():
    results=[]
    for url in TARGETS:
        results.append(probe_one(url))
        time.sleep(1.0)
    out=Path("research/market_fair_odds/source_probe_2026_10_18.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"status":"SOURCE_DISCOVERY_ONLY_NOT_ODDS_DATA",
                               "probed_at_utc":datetime.now(timezone.utc).isoformat(),
                               "pages":results},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"page_statuses":[{"url":x["url"],"status":x["status"],
                                        "links":len(x.get("relevant_links",[]))}
                                       for x in results]},ensure_ascii=False))


if __name__=="__main__":
    main()

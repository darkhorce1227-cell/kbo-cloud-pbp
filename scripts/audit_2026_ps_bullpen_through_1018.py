#!/usr/bin/env python3
"""Collect 2026 post-season KIA/LG exact pitcher workload through Oct 18.

Naver completed pitcher box lines for 10/14 WC, 10/17 & 10/18 준PO;
separately verify official PBP score audits for 10/18; do not infer
unavailable pitchers, future lineup, medical issues or pitch caps.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from kbo_pbp import naver
from collect_rs_starter_lines import parse_ip

ROOT=Path(__file__).resolve().parents[1]
SOURCE=[
    {"date":"2026-10-14","game_pk":"44441014OBLG02026","away":"OB","home":"LG","round":"WC"},
    {"date":"2026-10-17","game_pk":"33331017LGHT02026","away":"LG","home":"HT","round":"준PO1"},
    {"date":"2026-10-18","game_pk":"33331018LGHT02026","away":"LG","home":"HT","round":"준PO2"}
]
OUT=ROOT/"research/postseason_expected_ip/2026_shadow/ps_bullpen_workload_through_2026_10_18.json"


def parse_count(value):
    if value in (None,""):return None
    try:
        x=int(str(value).strip().replace(",",""))
        if 0<=x<=250:return x
    except (ValueError,TypeError):pass
    return None


def collect():
    audit=json.loads((ROOT/"state/postgame_score_audit_2026-10-18.json").read_text(encoding="utf-8"))
    if audit.get("validation")!="PASS" or audit.get("expected_game_ids")!=["33331018LGHT02026"]:
        raise ValueError("Official 10/18 PBP score crosscheck is not PASS")
    games=[]
    for game in SOURCE:
        relay=naver.fetch_relay(game["game_pk"],1)
        sides={}
        for side,team in (("away",game["away"]),("home",game["home"])):
            box=(relay.get(f"{side}Lineup") or {}).get("pitcher") or []
            if not box:raise ValueError(f"missing completed pitcher box for {game['game_pk']} {side}")
            starters=[x for x in box if int(x.get("seqno") or -1)==1]
            if len(starters)!=1:raise ValueError("Expected exactly one starter")
            players=[]
            for p in box:
                ident=str(p.get("pcode") or "")
                if not ident:raise ValueError("missing pitcher id")
                outs,ip=parse_ip(p.get("inn"))
                seq=int(p.get("seqno") or -1)
                if outs<0 or seq<1:raise ValueError("bad completed pitching record")
                players.append({
                  "id":ident,"name":str(p.get("name") or ""),
                  "seqno":seq,"role":"STARTER" if seq==1 else "RELIEF",
                  "innings_pitch":ip,"outs":outs,
                  "pitch_count":parse_count(p.get("ballCount")),
                  "inn_raw":str(p.get("inn")),
                })
            if not all(p["pitch_count"] is not None for p in players):
                raise ValueError(f"Cannot audit missing pitch counts game={game['date']} team={team}")
            totals={
                "all_pitchers":len(players),
                "starter_outs":sum(p["outs"] for p in players if p["role"]=="STARTER"),
                "starter_pitches":sum(p["pitch_count"] for p in players if p["role"]=="STARTER"),
                "relief_outs":sum(p["outs"] for p in players if p["role"]=="RELIEF"),
                "relief_pitches":sum(p["pitch_count"] for p in players if p["role"]=="RELIEF"),
                "reliever_appearances":sum(p["role"]=="RELIEF" for p in players),
            }
            sides[team]={"team":team,"side":side,"summary":totals,"pitchers":players}
        games.append({**game,"sides":sides})
    per_team={}
    for team in ("HT","LG"):
        appearance=[]
        for game in games:
            if team not in game["sides"]:continue
            for p in game["sides"][team]["pitchers"]:
                appearance.append({"date":game["date"],"round":game["round"],**p})
        relief=[x for x in appearance if x["role"]=="RELIEF"]
        rel_by_player=defaultdict(list)
        for x in relief:rel_by_player[(x["id"],x["name"])].append(x)
        per_team[team]={
            "relief_appearances":len(relief),
            "relief_pitch_count_total":sum(x["pitch_count"] for x in relief),
            "relief_outs_total":sum(x["outs"] for x in relief),
            "all_relievers":[
               {"player_id":pid,"name":name,"appearance_count":len(rr),
                "game_dates":[v["date"] for v in rr],
                "pitch_count_total":sum(v["pitch_count"] for v in rr),
                "last_appearance_date":max(v["date"] for v in rr),
                "pitch_counts_by_game":[{"date":v["date"],"pitches":v["pitch_count"],
                                         "outs":v["outs"]} for v in rr]}
               for (pid,name),rr in sorted(rel_by_player.items(),
                    key=lambda item:-sum(x["pitch_count"] for x in item[1]))],
        }
    report={
        "status":"OFFICIAL_NAVER_POSTSEASON_BOX_PITCH_COUNTS_VERIFIED",
        "cutoff_date":"2026-10-18",
        "next_game":"2026-10-20 HT@LG, 18:30 KST",
        "official_10_18_final_audit":"PASS",
        "games":games,
        "team_summaries":per_team,
        "limitations":[
          "Pitcher box totals are not a full pitch-tracking workload or measure of fatigue",
          "Rest day 10/19 does not imply unrestricted relief availability on 10/20",
          "Prior 10/14 LG WC relief appearances included; prior KIA games 10/14 not played",
          "Next game's activated relievers and bullpen role plans are not confirmed",
          "Source is postgame pitching box score, not inferred result from prior betting prices",
          "No changes to precommitted CLEAN PRE μ or expected IP forecast"
        ],
    }
    return report


def main():
    if OUT.exists():
        raise SystemExit("REFUSE_OVERWRITE_FIXED_POSTSEASON_WORKLOAD_RECORD")
    obj=collect()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf-8") as handle:
        json.dump(obj,handle,ensure_ascii=False,indent=2)
        handle.write("\n")
    print(json.dumps({k:{"relief_appearances":v["relief_appearances"],
                         "pitches":v["relief_pitch_count_total"],
                         "pitchers":v["all_relievers"][:5]}
                      for k,v in obj["team_summaries"].items()},ensure_ascii=False))


if __name__=="__main__":main()

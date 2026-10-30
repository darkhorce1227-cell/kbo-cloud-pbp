#!/usr/bin/env python3
"""Offline, fail-closed import of user-captured 2026-10-29 Samsung@KIA PBP.

Input: original raw ZIP produced on a user's computer against Naver's API.
Reads only its 9 relay JSONs and dated schedule JSON, verifies SHA-256 and
the explicit GAME_END, converts via kbo_pbp.statcast.game_frame, and merges
into the last validated rolling-release Parquet WITHOUT any Naver HTTP call.

A workflow retrieves the validated release from GitHub, runs this module,
independently applies the repo's existing continuity/date/score QA gates,
and publishes ONLY after all PASS. No retroactive betting forecast changes.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
from kbo_pbp import statcast

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"cloud"))
import update_pbp as core
import update_pbp_guarded as guarded
import update_pbp_integrity as integrity
import build_postgame_score_audit as score_audit

RAW_DIR=ROOT/"state/manual_pbp_raw"
RAW_PATTERN="kbo_20261029_SSHT_raw_*.zip"
BASE=ROOT/"out/baseline/kbo_pbp_2026_latest.parquet"
ID="55551029SSHT02026"
DAY=date(2026,10,27)
PINNED_ARCHIVE_SHA256="89657c00721c9588577ddd8ce07b9f2564039dbe90923401b64571084a1d21af"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract(raw: Path):
    with ZipFile(raw) as z:
        if z.testzip() is not None:
            raise RuntimeError("Damaged ZIP member")
        files={n.replace("\\","/"):n for n in z.namelist()}
        if len(files)!=len(z.namelist()):
            raise RuntimeError("Duplicate archive filename")
        required={"manifest.json","schedule.json"} | {
            f"relays/inning_{i:02d}.json" for i in range(1,10)}
        if set(files)!=required:
            raise RuntimeError(f"Unexpected archive members; missing={required-set(files)}")
        def read(name):
            return z.read(files[name])
        manifest=json.loads(read("manifest.json"))
        if manifest.get("schema")!="kbo_naver_raw_relay_capture_v1":
            raise RuntimeError("Unknown raw relay capture schema")
        if (manifest.get("game_id")!=ID or manifest.get("target_date_kst")!=DAY.isoformat()
                or manifest.get("inning_count")!=9 or
                manifest.get("game_end_verified_in_relay") is not True):
            raise RuntimeError("Wrong game/date or no terminal game proof")
        if manifest.get("schedule_sha256")!=_sha(read("schedule.json")):
            raise RuntimeError("Schedule SHA-256 does not match manifest")
        schedule=json.loads(read("schedule.json"))
        if schedule.get("success") is not True:
            raise RuntimeError("Unsuccessful Naver schedule response")
        found=[x for x in schedule.get("result",{}).get("games",[])
               if x.get("gameId")==ID]
        if len(found)!=1:
            raise RuntimeError("Exactly one actual game in schedule needed")
        game=found[0]
        expected={
            "gameDate":"2026-10-29","homeTeamCode":"HT","awayTeamCode":"SS",
            "homeTeamScore":4,"awayTeamScore":6,"statusCode":"RESULT",
            "statusNum":4,
        }
        for key,v in expected.items():
            if game.get(key)!=v:
                raise RuntimeError(f"Naver schedule {key} mismatch: {game.get(key)!r}")
        if game.get("statusInfo") not in ("9회말","9회초"):
            raise RuntimeError("Not a completed nine inning Naver game")
        rows=manifest.get("innings") or []
        if len(rows)!=9 or sorted(int(e.get("inning")) for e in rows)!=list(range(1,10)):
            raise RuntimeError("Nine uniquely numbered innings required")
        relays={}
        for row in rows:
            inning=int(row["inning"])
            name=f"relays/inning_{inning:02d}.json"
            if row.get("file")!=name:
                raise RuntimeError(f"Noncanonical manifest member: {row.get('file')}")
            raw_member=read(name)
            if len(raw_member)!=int(row.get("bytes",-1)) or _sha(raw_member)!=row.get("sha256"):
                raise RuntimeError(f"Tampered or incomplete inning {inning}")
            wrapper=json.loads(raw_member)
            if wrapper.get("success") is not True:
                raise RuntimeError(f"Naver relay failed inning {inning}")
            item=wrapper.get("result",{}).get("textRelayData")
            if not isinstance(item,dict) or item.get("gameId")!=ID:
                raise RuntimeError(f"Wrong game relay inning {inning}")
            found_terminal=any(
                str(ev.get("type"))=="99"
                for block in item.get("textRelays",[])
                for ev in (block.get("textOptions") or [])
            )
            if found_terminal!=(inning==9) or found_terminal!=bool(row.get("game_end_type_99")):
                raise RuntimeError(f"Mislabelled GAME_END at inning {inning}")
            relays[inning]=item
        # Cross-check completed ninth-bottom and final source scores.
        final=relays[9].get("inningScore",{})
        for side in ("away","home"):
            scores=list(final.get(side,{}).values())
            if not scores or any(str(v)!="-" and not str(v).isdigit() for v in scores):
                raise RuntimeError("Invalid inningScore member for "+side)
        if str(final["home"].get("9"))!="0":
            raise RuntimeError("Expected played scoreless 9th bottom for KIA")
        if sum(int(v) for v in final.get("away",{}).values() if str(v).isdigit())!=6:
            raise RuntimeError("Away score mismatch across source responses")
        if sum(int(v) for v in final.get("home",{}).values() if str(v).isdigit())!=4:
            raise RuntimeError("Home score mismatch across source responses")
        return relays,game,manifest


def run(raw=None,baseline=BASE):
    if raw is None:
        found=sorted(RAW_DIR.glob(RAW_PATTERN))
        if len(found)!=1:
            raise RuntimeError(f"Require exactly one original named capture ZIP, got {len(found)}")
        raw=found[0]
    if not raw.is_file() or not baseline.is_file():
        raise FileNotFoundError("Missing user ZIP or validated rolling release baseline")
    if _sha(raw.read_bytes()) != PINNED_ARCHIVE_SHA256:
        raise RuntimeError("Uploaded ZIP SHA-256 differs from independently verified chat original")
    relays,game,manifest=_extract(raw)
    new=statcast.game_frame(relays,statcast.meta_from_schedule(game))
    if new.empty or set(new["game_pk"].astype(str))!={ID}:
        raise RuntimeError("Imported PBP has empty or incorrect game rows")
    if set(new["home_team"].astype(str))!={"HT"} or set(new["away_team"].astype(str))!={"SS"}:
        raise RuntimeError("Wrong reconstructed teams")
    if set(pd.to_datetime(new["game_date"]).dt.date)!={DAY}:
        raise RuntimeError("Wrong game date in reconstructed statcast")
    if new.duplicated(["game_pk","at_bat_number","pitch_number"],keep=False).any():
        raise RuntimeError("New PBP duplicate pitch keys")
    old=pd.read_parquet(baseline)
    prior_index=guarded._load_index(2026)
    required=set(prior_index.get("game_pks",[]))
    if len(required)!=726 or int(prior_index.get("game_count",0))!=726:
        raise RuntimeError("Expected exactly 726 independently validated prior game IDs")
    present=set(old["game_pk"].astype(str))
    if len(present)!=726 or not required.issubset(present):
        raise RuntimeError("Rolling release lacks previously validated game IDs")
    if ID in present:
        raise RuntimeError("Target game already in baseline; refuse silent overwrite")
    candidate=pd.concat([old,new],ignore_index=True,sort=False)
    existing=set(candidate["game_pk"].astype(str))
    if not present.issubset(existing) or len(existing)!=len(present)+1:
        raise RuntimeError("Season game-PK continuity failure")
    ok,diag=integrity._validate_with_date_integrity(candidate,DAY,[ID],{})
    if not ok:
        raise RuntimeError("PBP integrity validation failed: "+
                           json.dumps(diag,ensure_ascii=False)[:2600])
    out_gz,out_pq,_=integrity._export_with_date_integrity(candidate,2026,DAY)
    status=core.Result(
        version="0.6.0-OFFLINE_ARCHIVE",
        state="UPDATED",validation="PASS",target_date=DAY.isoformat(),
        checked_at_kst=core.now_kst().isoformat(),
        first_check_kst="OFFLINE_CAPTURE_AFTER_FINISH",
        expected_games=1,final_games=1,expected_game_ids=[ID],
        found_game_ids=[ID],rows=len(candidate),columns=len(candidate.columns),
        duplicate_stable_keys=0,missing_columns=[],
        csv_gz=out_gz.name,parquet=out_pq.name,
        sha256_csv_gz=core.sha256(out_gz),sha256_parquet=core.sha256(out_pq),
        last_success_date=DAY.isoformat(),
        reason="[USER_ARCHIVE_OFFLINE_RECOVERY] Naver 9-inning relay ZIP SHA-256, explicit terminal GAME_END, 2026-10-29 schedule 6:4 and all cumulative integrity gates PASS.",
        should_publish=True,retry_next_hour=False)
    core.save_result(status)
    # This is a captured Naver schedule and relays, not an independent live
    # second request. Never imply that the source is wholly independent.
    score_audit.fetch_official_final_scores=lambda d,ids:{ID:{"away":6,"home":4}}
    audit=score_audit.build_audit(candidate,asdict(status),out_pq)
    audit["offline_source_provenance"]="USER_UPLOADED_NAVER_ORIGINAL_RELAY_ZIP"
    audit["source_raw_zip_sha256"]=_sha(raw.read_bytes())
    audit["official_final_crosscheck_provenance"]="NAVER_SCHEDULE_JSON_FROM_SAME_USER_ARCHIVE_NOT_AN_INDEPENDENT_SECOND_FETCH"
    audit["original_capture_utc"]=manifest["captured_at_utc"]
    if audit.get("validation")!="PASS":
        raise RuntimeError("Failed PBP-derived F5/final score audit: "+
                           json.dumps(audit,ensure_ascii=False)[:1800])
    scores=audit["games"]
    if len(scores)!=1 or scores[0].get("status")!="PASS" or scores[0]["final"]["away"]!=6 or scores[0]["final"]["home"]!=4:
        raise RuntimeError("Expected final Samsung 6 - KIA 4 audit failed")
    score_audit._write(audit,DAY)
    report={"status":"OFFLINE_PBP_MERGE_VALIDATED",
      "target_date":DAY.isoformat(),"game_pk":ID,
      "previous_game_count":len(present),"new_game_count":len(existing),
      "new_pitch_rows":len(new),"total_pitch_rows":len(candidate),
      "original_raw_zip_sha256":_sha(raw.read_bytes()),
      "original_naver_capture_utc":manifest["captured_at_utc"],
      "game_end_inning":9,"final_away":6,"final_home":4,
      "binary_derived_f5_and_final":scores[0],
      "sha256_parquet":status.sha256_parquet,
      "validated_release_file":out_pq.name,
      "upstream_naver_api_used_by_importer":False,
      "legacy_game_pk_continuity":"PASS",
      "audited_score_source":"Naver schedule inside same raw capture; not independent external check"}
    path=ROOT/"state/manual_pbp_raw/2026-10-29_SSHT_import_audit.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    run()

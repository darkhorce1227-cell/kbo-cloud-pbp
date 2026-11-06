#!/usr/bin/env python3
"""Import auto-captured original Naver PBP ZIPs offline and fail-closed.

Uses the existing validated rolling Parquet as baseline, never Naver HTTP.
Includes independent-of-capture binary-derived F5/final checks, but note that
the Naver schedule included in each ZIP is NOT an independent external source.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import pandas as pd
from kbo_pbp import statcast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
import update_pbp as core
import update_pbp_guarded as guarded
import update_pbp_integrity as integrity
import build_postgame_score_audit as score_audit

RAW_DIR = ROOT / "state/auto_pbp_raw"
BASE = ROOT / "out/baseline/kbo_pbp_2026_latest.parquet"
RESULT_PATH = ROOT / "out/auto_import_result.json"
ID_RE = re.compile(r"^[0-9]{8}[A-Z]{4}0[0-9]{4}$")
MAX_INNINGS = 15


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_archive(path: Path) -> tuple[dict, dict, dict[int, dict], dict]:
    """Validate source-file bytes, complete inning set and explicit terminal."""
    if not path.is_file() or path.stat().st_size > 30_000_000:
        raise ValueError(f"Missing or oversized source ZIP: {path}")
    with ZipFile(path) as z:
        members = z.infolist()
        if len(members) < 7 or len(members) > 2 + MAX_INNINGS:
            raise ValueError(f"Unexpected member count in {path}")
        if z.testzip() is not None:
            raise ValueError("ZIP member integrity failure")
        normalized = {entry.filename.replace("\\", "/"): entry for entry in members}
        if len(normalized) != len(members):
            raise ValueError("Duplicate ZIP members")
        if any(entry.file_size > 15_000_000 for entry in members):
            raise ValueError("Oversized uncompressed source member")
        def read(name: str) -> bytes:
            if name not in normalized:
                raise ValueError(f"Missing source ZIP member: {name}")
            return z.read(normalized[name])
        manifest = json.loads(read("manifest.json"))
        if manifest.get("schema") != "kbo_naver_raw_relay_capture_v1":
            raise ValueError("Unsupported source schema")
        gid = str(manifest.get("game_id") or "")
        day = date.fromisoformat(str(manifest["target_date_kst"]))
        if not ID_RE.fullmatch(gid) or not gid.endswith(str(day.year)) or gid[4:8] != day.strftime("%m%d"):
            raise ValueError(f"Game identity/date mismatch: {gid} / {day}")
        inn_count = manifest.get("inning_count")
        if not isinstance(inn_count, int) or not 5 <= inn_count <= MAX_INNINGS:
            raise ValueError(f"Invalid terminal inning count: {inn_count}")
        expected = {"manifest.json", "schedule.json"} | {f"relays/inning_{n:02d}.json" for n in range(1, inn_count + 1)}
        if set(normalized) != expected:
            raise ValueError(f"Unexpected/missing ZIP members: {sorted(set(normalized)^expected)}")
        if sha(read("schedule.json")) != manifest.get("schedule_sha256"):
            raise ValueError("Schedule SHA256 mismatch")
        schedule = json.loads(read("schedule.json"))
        if schedule.get("success") is not True or not isinstance((schedule.get("result") or {}).get("games"), list):
            raise ValueError("No verified Naver date schedule")
        games = [g for g in schedule["result"]["games"] if g.get("gameId") == gid]
        if len(games) != 1:
            raise ValueError(f"Expected exactly one schedule game ID: {gid}")
        game = games[0]
        if (game.get("gameDate") != day.isoformat() or
                game.get("statusCode") != "RESULT" or game.get("statusNum") != 4 or
                str(game.get("awayTeamCode")) != gid[8:10] or
                str(game.get("homeTeamCode")) != gid[10:12]):
            raise ValueError("Naver game identity/final-status mismatch")
        ending = re.fullmatch(r"(\d+)회(초|말)", str(game.get("statusInfo") or ""))
        if not ending or int(ending.group(1)) != inn_count:
            raise ValueError("Schedule terminal inning/half mismatch")
        if manifest.get("game_end_verified_in_relay") is not True:
            raise ValueError("Original capture lacks terminal proof")
        row_specs = manifest.get("innings")
        if (not isinstance(row_specs, list) or len(row_specs) != inn_count or
                sorted(r.get("inning") for r in row_specs) != list(range(1, inn_count + 1))):
            raise ValueError("Incomplete/duplicate inning manifest")
        relays = {}
        for row in row_specs:
            number = row["inning"]
            name = f"relays/inning_{number:02d}.json"
            if row.get("file") != name:
                raise ValueError("Manifest noncanonical filename")
            raw = read(name)
            if len(raw) != row.get("bytes") or sha(raw) != row.get("sha256"):
                raise ValueError(f"Source relay SHA/size mismatch at inning {number}")
            wrapper = json.loads(raw)
            item = (wrapper.get("result") or {}).get("textRelayData")
            if wrapper.get("success") is not True or not isinstance(item, dict) or item.get("gameId") != gid:
                raise ValueError(f"Wrong/incomplete inning relay {number}")
            if not isinstance(item.get("textRelays"), list):
                raise ValueError(f"Missing inning {number} events")
            has_end = any(str(opt.get("type")) == "99"
                          for block in (item.get("textRelays") or [])
                          for opt in (block.get("textOptions") or []))
            if has_end != (number == inn_count) or has_end != row.get("game_end_type_99"):
                raise ValueError(f"Unexpected GAME_END terminal proof at inning {number}")
            relays[number] = item
        final = relays[inn_count].get("inningScore")
        if not isinstance(final, dict):
            raise ValueError("No final original inningScore")
        f5 = {}
        for side in ("away", "home"):
            scores = final.get(side)
            if not isinstance(scores, dict) or set(scores) != {str(i) for i in range(1, inn_count + 1)}:
                raise ValueError(f"Missing/extra inningScore keys for {side}")
            if any(str(s) != "-" and not str(s).isdigit() for s in scores.values()):
                raise ValueError(f"Invalid inningScore value for {side}")
            if sum(int(s) for s in scores.values() if str(s).isdigit()) != int(game[f"{side}TeamScore"]):
                raise ValueError(f"Final score mismatch for {side}")
            start5 = [str(scores[str(i)]) for i in range(1, 6)]
            if any(not s.isdigit() for s in start5):
                raise ValueError(f"F5 incomplete for {side}")
            f5[side] = sum(int(s) for s in start5)
        if str(final["away"][str(inn_count)]) == "-":
            raise ValueError("Terminal away half not played")
        if ending.group(2) == "초" and str(final["home"][str(inn_count)]) != "-":
            raise ValueError("Bottom played, yet Naver says top ended")
        if ending.group(2) == "말" and str(final["home"][str(inn_count)]) == "-":
            raise ValueError("Bottom not played, yet Naver says bottom ended")
        record = {
            "game_pk": gid, "day": day.isoformat(), "source": path.name,
            "source_sha256": sha(path.read_bytes()), "terminal_inning": inn_count,
            "terminal_half": "top" if ending.group(2) == "초" else "bottom",
            "f5": f5,
            "final": {"away": int(game["awayTeamScore"]), "home": int(game["homeTeamScore"])},
            "schedule": schedule,
        }
        return record, game, relays, manifest


def verify_only(path: Path) -> None:
    record, game, relays, _ = get_archive(path)
    frame = statcast.game_frame(relays, statcast.meta_from_schedule(game))
    if frame.empty or set(frame["game_pk"].astype(str)) != {record["game_pk"]}:
        raise ValueError("Statcast parser gave empty/wrong game")
    if len(frame) < 50 or frame.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
        raise ValueError("Too few/duplicate stable PBP keys")
    if set(pd.to_datetime(frame["game_date"]).dt.date) != {date.fromisoformat(record["day"])}:
        raise ValueError("Statcast game-date mismatch")
    print(json.dumps({"result": "ARCHIVE_VERIFY_PASS", "game": record["game_pk"],
                      "rows": len(frame), "f5": record["f5"], "final": record["final"],
                      "terminal_inning": record["terminal_inning"]}, ensure_ascii=False))


def run() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW_DIR.glob("kbo_*_raw_*.zip"))
    if not paths:
        return {"publish": False, "reason": "NO_AUTO_ARCHIVES"}
    if not BASE.is_file():
        raise FileNotFoundError(f"Baseline release Parquet missing: {BASE}")
    before = pd.read_parquet(BASE)
    if before.empty:
        raise ValueError("Refusing empty original baseline")
    old_index = guarded._load_index(2026)
    expected = set(old_index.get("game_pks", []))
    original = set(before["game_pk"].astype(str))
    if (len(expected) != int(old_index.get("game_count", -1)) or expected != original):
        raise ValueError("Original release and validated game index are inconsistent")
    status = json.loads((ROOT / "state/status.json").read_text(encoding="utf-8"))
    if (int(status["rows"]) != len(before) or status["validation"] != "PASS" or
            core.sha256(BASE) != status["sha256_parquet"]):
        raise ValueError("Original release Parquet/status mismatch")
    records: dict[str, tuple[dict, dict, dict[int, dict]]] = {}
    for path in paths:
        record, game, relays, _ = get_archive(path)
        gid = record["game_pk"]
        if gid in records:
            raise ValueError(f"Duplicate captured game ID: {gid}")
        records[gid] = (record, game, relays)
    new_records = [x for x in records.values() if x[0]["game_pk"] not in original]
    if not new_records:
        return {"publish": False, "reason": "ALL_AUTO_GAMES_ALREADY_VALIDATED"}
    grouped: dict[str, list[tuple[dict, dict, dict[int, dict]]]] = {}
    for entry in new_records:
        grouped.setdefault(entry[0]["day"], []).append(entry)
    history = []
    candidate = before
    for day_s in sorted(grouped):
        group = grouped[day_s]
        day = date.fromisoformat(day_s)
        # A complete date-scoped schedule is retained in EVERY original ZIP.
        schedule_games = group[0][0]["schedule"]["result"]["games"]
        finals = {str(g.get("gameId")) for g in schedule_games
                  if isinstance(g, dict) and g.get("gameDate") == day_s
                  and ID_RE.fullmatch(str(g.get("gameId") or ""))
                  and g.get("statusCode") == "RESULT" and g.get("statusNum") == 4}
        received = original | {r[0]["game_pk"] for r in new_records if r[0]["day"] == day_s}
        if not finals or not finals.issubset(received):
            raise ValueError(f"Missing completed game archive on {day_s}: {sorted(finals-received)}")
        frames = []
        ids = []
        for record, game, relays in group:
            gid = record["game_pk"]
            if gid in set(candidate["game_pk"].astype(str)):
                raise ValueError("Attempted duplicate game import")
            frame = statcast.game_frame(relays, statcast.meta_from_schedule(game))
            if frame.empty or set(frame["game_pk"].astype(str)) != {gid}:
                raise ValueError(f"Wrong/empty Naver parsed frame: {gid}")
            if len(frame) < 50 or frame.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
                raise ValueError(f"Duplicate/stunted PBP: {gid}")
            if set(pd.to_datetime(frame["game_date"]).dt.date) != {day}:
                raise ValueError(f"Wrong parsed game date: {gid}")
            if (set(frame["home_team"].astype(str)) != {str(game["homeTeamCode"])} or
                    set(frame["away_team"].astype(str)) != {str(game["awayTeamCode"])}):
                raise ValueError(f"Wrong parsed teams: {gid}")
            frames.append(frame)
            ids.append(gid)
        before_ids = set(candidate["game_pk"].astype(str))
        candidate = pd.concat([candidate, *frames], ignore_index=True, sort=False)
        after_ids = set(candidate["game_pk"].astype(str))
        if after_ids != before_ids | set(ids) or len(after_ids) != len(before_ids) + len(ids):
            raise ValueError("Legacy game PK continuity lost")
        ok, diag = integrity._validate_with_date_integrity(candidate, day, ids, {})
        if not ok:
            raise RuntimeError("Season PBP validation failure: " + json.dumps(diag, ensure_ascii=False)[:2200])
        out_gz, out_pq, _ = integrity._export_with_date_integrity(candidate, 2026, day)
        result = core.Result(
            version="0.7.0-AUTO_OFFLINE_ZIP", state="UPDATED", validation="PASS",
            target_date=day_s, checked_at_kst=core.now_kst().isoformat(),
            first_check_kst="LOCAL_NETWORK_CAPTURE_AFTER_FINAL",
            expected_games=len(ids), final_games=len(ids),
            expected_game_ids=ids, found_game_ids=ids,
            rows=len(candidate), columns=len(candidate.columns),
            duplicate_stable_keys=0, missing_columns=[],
            csv_gz=out_gz.name, parquet=out_pq.name,
            sha256_csv_gz=core.sha256(out_gz), sha256_parquet=core.sha256(out_pq),
            last_success_date=day_s,
            reason="[AUTO_OFFLINE] Original local-network Naver relays with terminal type99, SHA256, score and continuity PASS",
            should_publish=True, retry_next_hour=False,
        )
        core.save_result(result)
        official_scores = {r[0]["game_pk"]: r[0]["final"] for r in group}
        score_audit.fetch_official_final_scores = lambda d, game_ids, values=official_scores: values
        audit = score_audit.build_audit(candidate, asdict(result), out_pq)
        if audit.get("validation") != "PASS" or len(audit.get("games", [])) != len(ids):
            raise RuntimeError("PBP-derived final/F5 audit failed: " + json.dumps(audit, ensure_ascii=False)[:2200])
        by_pk = {r["game_pk"]: r for r in audit["games"]}
        for source_record, _, _ in group:
            actual = by_pk[source_record["game_pk"]]
            if (actual.get("status") != "PASS" or actual.get("f5", {}).get("away") != source_record["f5"]["away"]
                    or actual.get("f5", {}).get("home") != source_record["f5"]["home"]
                    or actual.get("final", {}).get("away") != source_record["final"]["away"]
                    or actual.get("final", {}).get("home") != source_record["final"]["home"]
                    or actual.get("final", {}).get("inning") != source_record["terminal_inning"]
                    or str(actual.get("final", {}).get("half", "")).lower() != source_record["terminal_half"]):
                raise RuntimeError(f"Original/PBP-derived score/terminal mismatch: {source_record['game_pk']}")
        audit["source_schedule_provenance"] = "same Naver capture archive, NOT independent second request"
        score_audit._write(audit, day)
        detail = {
            "status": "AUTO_PBP_MERGE_VALIDATED", "target_date": day_s,
            "prior_games": len(before_ids), "new_games": len(after_ids),
            "prior_rows": len(candidate) - sum(len(f) for f in frames),
            "new_rows": sum(len(f) for f in frames),
            "total_rows": len(candidate), "game_pks": ids,
            "source_archives": [{k: rec[k] for k in
                                 ("game_pk", "source", "source_sha256", "terminal_inning", "f5", "final")}
                                for rec, _, _ in group],
            "sha256_parquet": core.sha256(out_pq),
            "source_score_crosscheck": "Naver schedule inside same original archive, not independent external",
        }
        path = RAW_DIR / f"import_audit_{day_s}.json"
        path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history.append(detail)
    return {"publish": True, "new_game_count": len(new_records),
            "old_game_count": len(original), "total_game_count": len(set(candidate["game_pk"].astype(str))),
            "dates": [x["target_date"] for x in history],
            "sha256_parquet": core.sha256(ROOT / "out/kbo_pbp_2026_latest.parquet"),
            "history": history}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify-zip", help="Verify one existing ZIP and parse PBP without publishing")
    opts = p.parse_args()
    try:
        if opts.verify_zip:
            verify_only(Path(opts.verify_zip))
            return 0
        result = run()
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"AUTO_IMPORT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

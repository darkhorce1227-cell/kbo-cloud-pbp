#!/usr/bin/env python3
"""Offline 2026-10-26 KIA@Samsung PBP recovery from user-collected Naver JSON.

This script NEVER contacts Naver. Input is one SHA-pinned raw capture ZIP
uploaded to the GitHub repository. Preserve validated old-game PKs, reconstruct
the game through the official kbo_pbp event parser, and fail closed unless the
combined Parquet scores, F5 and ALL current historical game PKs pass audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
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

GAME_ID = "55551026HTSS02026"
TARGET = date(2026, 10, 26)
YEAR = 2026
ZIP_SHA256 = "abb9ba6656cd23ea80f18b8817d5da49f775cf317367608a345c2fe86ab7aec2"
ZIP_PATH = ROOT / "imports" / "kbo_20261026_HTSS_raw_20261027_152005.zip"
PREVIOUS_PARQUET = ROOT / "previous_release" / "kbo_pbp_2026_latest.parquet"
SOURCE_SCHEDULE_SHA = "03da2e8e6aa67fe47aa610d84e3ff4c7473f7be28af4b1e4a5ae5eedc8113554"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_original_zip(path: Path) -> tuple[dict, dict, dict[int, dict]]:
    if not path.is_file() or sha256_bytes(path.read_bytes()) != ZIP_SHA256:
        raise ValueError("INPUT_ZIP_SHA256_MISMATCH or missing uploaded original")

    with ZipFile(path) as archive:
        names = {entry.filename.replace("\\", "/"): entry for entry in archive.infolist()}
        expected = {"manifest.json", "schedule.json"} | {
            f"relays/inning_{i:02d}.json" for i in range(1, 10)
        }
        if set(names) != expected or len(archive.infolist()) != len(expected):
            raise ValueError("ZIP_MISSING_OR_UNEXPECTED_MEMBERS")
        if any(x.file_size > 400_000 for x in names.values()):
            raise ValueError("ZIP_MEMBER_UNEXPECTEDLY_LARGE")
        manifest = json.loads(archive.read(names["manifest.json"]))
        raw_schedule = archive.read(names["schedule.json"])
        if sha256_bytes(raw_schedule) != SOURCE_SCHEDULE_SHA:
            raise ValueError("SCHEDULE_SHA256_MISMATCH")
        schedule = json.loads(raw_schedule)
        if manifest.get("schema") != "kbo_naver_raw_relay_capture_v1":
            raise ValueError("UNSUPPORTED_RAW_RELAY_CAPTURE_SCHEMA")
        if manifest.get("game_id") != GAME_ID or manifest.get("target_date_kst") != str(TARGET):
            raise ValueError("WRONG_GAME_OR_DATE")
        if manifest.get("schedule_sha256") != SOURCE_SCHEDULE_SHA:
            raise ValueError("SCHEDULE_PROVENANCE_MISMATCH")
        if manifest.get("schedule_status_num") != 4 or manifest.get("inning_count") != 9:
            raise ValueError("POSTPONED_9_INNINGS_METADATA_MISMATCH")
        if manifest.get("game_end_verified_in_relay") is not True:
            raise ValueError("MISSING_CAPTURE_GAME_END_CLAIM")

        items = schedule.get("result", {}).get("games") or []
        official = [g for g in items if g.get("gameId") == GAME_ID]
        if len(official) != 1:
            raise ValueError("OFFICIAL_SCHEDULE_GAME_NOT_UNIQUE")
        entry = official[0]
        if any((
            entry.get("gameDate") != str(TARGET),
            entry.get("categoryId") != "kbo",
            entry.get("homeTeamCode") != "SS",
            entry.get("awayTeamCode") != "HT",
            entry.get("homeTeamScore") != 1,
            entry.get("awayTeamScore") != 5,
            entry.get("statusNum") != 4,
            entry.get("statusCode") != "RESULT",
            entry.get("cancel") is not False,
        )):
            raise ValueError("OFFICIAL_COMPLETED_GAME_METADATA_MISMATCH")

        indexed = manifest.get("innings") or []
        if [x.get("inning") for x in indexed] != list(range(1, 10)):
            raise ValueError("INNINGS_NOT_ONE_THROUGH_NINE")
        relays = {}
        for row in indexed:
            inn = row["inning"]
            member = f"relays/inning_{inn:02d}.json"
            if row.get("file") != member:
                raise ValueError("MANIFEST_INNING_PATH_MISMATCH")
            data = archive.read(names[member])
            if row.get("bytes") != len(data) or row.get("sha256") != sha256_bytes(data):
                raise ValueError("ORIGINAL_RELAY_MEMBER_SHA256_MISMATCH")
            wrapper = json.loads(data)
            if str(wrapper.get("code")) != "200" or wrapper.get("success") is not True:
                raise ValueError("NAVER_RELAY_HTTP_OR_RESULT_NOT_SUCCESS")
            relay = wrapper.get("result", {}).get("textRelayData")
            if not isinstance(relay, dict) or relay.get("gameId") != GAME_ID:
                raise ValueError("NAVER_RELAY_WRONG_GAME")
            if not isinstance(relay.get("textRelays"), list):
                raise ValueError("NAVER_RELAY_EVENTS_MISSING")
            ends = [e for block in relay["textRelays"]
                    for e in block.get("textOptions") or []
                    if int(e.get("type", -1)) == 99]
            if bool(ends) != (inn == 9):
                raise ValueError("GAME_END_WRONG_INNING_OR_MISSING")
            if ends:
                terminal = ends[-1].get("currentGameState", {})
                if (int(terminal.get("awayScore", -1)),
                    int(terminal.get("homeScore", -1))) != (5, 1):
                    raise ValueError("RELAY_FINAL_SCORE_DISAGREES_WITH_SCHEDULE")
            if bool(row.get("game_end_type_99")) != bool(ends):
                raise ValueError("MANIFEST_GAME_END_MISMATCH")
            relays[inn] = relay
    return manifest, entry, relays


def load_prior(prior_path: Path) -> tuple[pd.DataFrame, dict]:
    idx = json.loads((ROOT / "state" / f"validated_game_pks_{YEAR}.json").read_text(encoding="utf-8"))
    required = list(map(str, idx.get("game_pks") or []))
    if idx.get("game_count") != 724 or len(required) != 724 or len(set(required)) != 724:
        raise ValueError("EXPECTED_724_PRIOR_GAME_IDS_NOT_PRESENT")
    if guarded._digest(required) != idx.get("game_pk_sha256"):
        raise ValueError("PRIOR_GAME_ID_INDEX_HASH_MISMATCH")
    if GAME_ID in required:
        raise ValueError("2026_10_26_GAME_ALREADY_IN_PRIOR_INDEX")
    if not prior_path.is_file():
        raise ValueError("PRIOR_LATEST_RELEASE_PARQUET_MISSING")
    prior = pd.read_parquet(prior_path)
    if "game_pk" not in prior.columns:
        raise ValueError("PRIOR_PARQUET_MISSING_GAME_PK")
    unique = set(prior["game_pk"].dropna().astype(str).unique())
    if unique != set(required):
        raise ValueError("PRIOR_RELEASE_DOES_NOT_MATCH_724_GAME_VALIDATED_INDEX")
    if prior.duplicated(["game_pk", "at_bat_number", "pitch_number"], keep=False).any():
        raise ValueError("PRIOR_RELEASE_STABLE_KEY_DUPLICATES")
    return prior, idx


def run(raw_zip: Path, old_parquet: Path) -> dict:
    manifest, schedule, relays = read_original_zip(raw_zip)
    original, idx = load_prior(old_parquet)

    meta = statcast.meta_from_schedule(schedule)
    new = statcast.game_frame(relays, meta)
    if new.empty or len(new) < 100:
        raise ValueError("NEW_GAME_UNEXPECTEDLY_EMPTY")
    if core.REQUIRED_COLUMNS - set(new.columns):
        raise ValueError(f"NEW_GAME_MISSING_FIELDS:{sorted(core.REQUIRED_COLUMNS-set(new.columns))}")
    if new["game_pk"].astype(str).nunique() != 1 or str(new["game_pk"].iloc[0]) != GAME_ID:
        raise ValueError("DERIVED_GAME_ID_MISMATCH")
    if set(pd.to_datetime(new["game_date"]).dt.strftime("%Y-%m-%d")) != {str(TARGET)}:
        raise ValueError("DERIVED_GAME_DATE_MISMATCH")
    if set(new["away_team"].astype(str)) != {"HT"} or set(new["home_team"].astype(str)) != {"SS"}:
        raise ValueError("DERIVED_HOME_AWAY_MISMATCH")
    if set(pd.to_numeric(new["inning"]).astype(int)) != set(range(1, 10)):
        raise ValueError("DERIVED_INNINGS_NOT_ONE_THROUGH_NINE")
    if new.duplicated(["game_pk", "at_bat_number", "pitch_number"], keep=False).any():
        raise ValueError("DERIVED_PITCH_STABLE_KEYS_NOT_UNIQUE")

    all_columns = list(dict.fromkeys([*original.columns, *new.columns]))
    df = pd.concat([original.reindex(columns=all_columns),
                    new.reindex(columns=all_columns)], ignore_index=True)
    observed_ids = set(df["game_pk"].astype(str))
    if observed_ids != (set(idx["game_pks"]) | {GAME_ID}):
        raise ValueError("HISTORIC_GAME_CONTINUITY_REGRESSION")
    if len(observed_ids) != 725:
        raise ValueError("725_GAME_FINAL_CARDINALITY_MISMATCH")
    if len(df) != len(original) + len(new):
        raise ValueError("DATASET_ROW_REGRESSION")
    if integrity._date_mismatches(df, YEAR):
        raise ValueError("GAME_PK_ENCODED_DATE_MISMATCH")
    ok, diag = core.validate_candidate(df, TARGET, [GAME_ID], {})
    if not ok:
        raise ValueError("CANDIDATE_SCHEMA_OR_STABLE_KEY_VALIDATION_FAILED: "
                         + json.dumps(diag, ensure_ascii=False))

    csv_gz, parquet, manifest_path = guarded._guarded_export(df, YEAR, TARGET)
    result = core.Result(
        version="2026-10-26-offline-raw-PBP-recovery-v1",
        state="UPDATED",
        validation="PASS",
        target_date=str(TARGET),
        checked_at_kst=core.now_kst().isoformat(),
        first_check_kst="17:00",
        expected_games=1,
        final_games=1,
        expected_game_ids=[GAME_ID],
        found_game_ids=[GAME_ID],
        rows=len(df),
        columns=len(df.columns),
        duplicate_stable_keys=0,
        missing_columns=[],
        csv_gz=csv_gz.name,
        parquet=parquet.name,
        sha256_csv_gz=core.sha256(csv_gz),
        sha256_parquet=core.sha256(parquet),
        last_success_date=str(TARGET),
        reason=("[OFFLINE_USER_ORIGINAL_NAVER_CAPTURE] all nine relay "
                "innings SHA-256 checked; explicit type-99 GAME_END; "
                "official result KIA 5 Samsung 1; all 724 prior game IDs "
                "preserved; reconstructed with original kbo_pbp.statcast."),
        should_publish=True,
        retry_next_hour=False,
    )

    original_fetch = score_audit.fetch_official_final_scores
    try:
        # Offline authoritative *uploaded original* Naver schedule, not a
        # fabricated score or a live request from GitHub Actions.
        score_audit.fetch_official_final_scores = (
            lambda target, ids: {GAME_ID: {"away": 5, "home": 1}}
        )
        binary_audit = score_audit.build_audit(df, vars(result), parquet)
    finally:
        score_audit.fetch_official_final_scores = original_fetch
    if binary_audit["validation"] != "PASS":
        raise ValueError("BINARY_DERIVED_POSTGAME_SCORE_AUDIT_FAIL:"
                         + json.dumps(binary_audit, ensure_ascii=False)[:2200])
    if (binary_audit.get("official_crosscheck_games_found") != 1
            or binary_audit["games"][0]["official_final_crosscheck"] != "PASS"
            or binary_audit["games"][0]["final"]["away"] != 5
            or binary_audit["games"][0]["final"]["home"] != 1
            or binary_audit["games"][0]["final"]["inning"] != 9):
        raise ValueError("VALIDATED_SCORE_CROSSCHECK_NOT_5_TO_1_IN_NINTH")

    binary_audit["offline_official_crosscheck_source"] = {
        "name": "schedule.json included in user collected original Naver API ZIP",
        "schedule_sha256": SOURCE_SCHEDULE_SHA,
        "raw_zip_sha256": ZIP_SHA256,
        "network_requests_to_naver": 0,
    }
    score_audit._write(binary_audit, TARGET)
    core.save_result(result)

    # The queued 10/26 date will no longer block subsequent genuine game days.
    queue = ROOT / "state" / "pbp_recovery_targets.json"
    if queue.is_file():
        state = json.loads(queue.read_text(encoding="utf-8"))
        for item in state.get("targets", []):
            if item.get("target_date") == str(TARGET):
                item["data_ingested"] = True
                item["validated_game_pk"] = GAME_ID
                item["validated_parquet_sha256"] = result.sha256_parquet
                item["validated_audit"] = "state/postgame_score_audit_2026-10-26.json"
        queue.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    summary = {
        "status": "PASS_BINARY_VALIDATED_OFFLINE_PBP_725_GAMES",
        "target_date": str(TARGET),
        "game_pk": GAME_ID,
        "source_zip_sha256": ZIP_SHA256,
        "previous_game_count": 724,
        "total_game_count": len(observed_ids),
        "new_game_pitch_rows": len(new),
        "old_dataset_rows": len(original),
        "new_dataset_rows": len(df),
        "parquet_sha256": result.sha256_parquet,
        "csv_gz_sha256": result.sha256_csv_gz,
        "score_audit_validation": binary_audit["validation"],
        "final_away": 5,
        "final_home": 1,
        "first_five": binary_audit["games"][0].get("f5"),
    }
    output = ROOT / "research" / "postseason_expected_ip" / "2026_shadow" / "offline_recovery_20261026_qa.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=ZIP_PATH)
    p.add_argument("--prior", type=Path, default=PREVIOUS_PARQUET)
    args = p.parse_args()
    run(args.zip, args.prior)


if __name__ == "__main__":
    main()

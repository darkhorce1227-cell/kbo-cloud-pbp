#!/usr/bin/env python3
"""Offline replay of real 12-inning KS2 source through the NEW local capture path."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))
import auto_capture_naver as capture
import auto_import_naver as importer

root = Path(__file__).resolve().parents[1]
fixture = root / "state/manual_pbp_raw/kbo_20261104_SSKT_raw_20261106_145433.zip"
with ZipFile(fixture) as zip_file:
    raw_schedule = zip_file.read("schedule.json")
    schedule = json.loads(raw_schedule)
    game = next(g for g in schedule["result"]["games"] if g.get("gameId") == "77771104SSKT02026")
    assert game["statusInfo"] == "12회말"
    def replay(session, url, params=None):
        if "relay" not in url:
            raise AssertionError(f"Unexpected fixture fetch {url}")
        inning = int(params["inning"])
        raw = zip_file.read(f"relays\\inning_{inning:02d}.json")
        return raw, json.loads(raw)

    old_function = capture.request_json
    old_dir = capture.RAW_DIR
    try:
        with tempfile.TemporaryDirectory() as folder:
            capture.RAW_DIR = Path(folder)
            capture.request_json = replay
            produced = capture.capture_one(None, "2026-11-04", raw_schedule, game)
            record, parsed_game, relays, manifest = importer.get_archive(produced)
            assert record["game_pk"] == "77771104SSKT02026"
            assert record["f5"] == {"away": 1, "home": 1}
            assert record["final"] == {"away": 3, "home": 2}
            assert manifest["inning_count"] == 12 and len(relays) == 12
            importer.verify_only(produced)
    finally:
        capture.request_json = old_function
        capture.RAW_DIR = old_dir
print("REAL_KS2_CAPTURE_AND_OFFLINE_VERIFICATION_PASS")

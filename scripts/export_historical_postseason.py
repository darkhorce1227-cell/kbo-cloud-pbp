#!/usr/bin/env python3
"""Validate cached postseason relays and export one row per pitch.

Run after historical_postseason.py. Uses the upstream event/PA parser rather
than attempting to reconstruct pitches from commentary text.
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from kbo_pbp import statcast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", default="historical_ps")
    args = parser.parse_args()
    root = Path(args.output) / str(args.year)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("errors"):
        raise RuntimeError("Raw collection manifest is not PASS")
    schedule = pd.read_csv(root / "schedule.csv", dtype={"gameId": str})
    if schedule.empty or not manifest.get("games"):
        raise RuntimeError("Empty schedule or manifest")
    if len(schedule) != len(manifest["games"]):
        raise RuntimeError("Schedule/manifest game count mismatch")
    indexed = schedule.set_index("gameId", drop=False)
    frames = []
    checks = []
    for entry in manifest["games"]:
        gid = str(entry["game_id"])
        innings = int(entry["innings"])
        if gid not in indexed.index or entry["status"] != "PASS" or entry["fetched"] != innings:
            raise RuntimeError(f"Manifest mismatch: {gid}")
        game = {}
        for inning in range(1, innings + 1):
            path = root / "raw" / f"{gid}_inning_{inning:02}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            # The upstream parser consumes the inner relay, not the API wrapper.
            relay = payload.get("result", {}).get("textRelayData", payload)
            if not isinstance(relay, dict) or not isinstance(relay.get("textRelays"), list):
                raise RuntimeError(f"Unexpected relay structure: {gid} inning {inning}")
            game[inning] = relay
        meta = statcast.meta_from_schedule(indexed.loc[gid])
        frame = statcast.game_frame(game, meta)
        if frame.empty:
            raise RuntimeError(f"No pitch rows parsed: {gid}")
        frame["season"] = args.year
        frame["round_code"] = indexed.loc[gid].get("roundCode")
        frames.append(frame)
        checks.append({"game_id": gid, "innings": innings, "rows": len(frame),
                       "unique_pa": frame["at_bat_number"].nunique()})
    data = pd.concat(frames, ignore_index=True)
    if len(data) != sum(c["rows"] for c in checks):
        raise RuntimeError("Row count mismatch")
    if data[["game_pk", "inning", "at_bat_number", "pitch_number"]].isna().any().any():
        raise RuntimeError("Missing essential pitch keys")
    data.to_csv(root / "postseason_pitches.csv.gz", index=False, compression="gzip")
    data.to_parquet(root / "postseason_pitches.parquet", index=False)
    report = {"year": args.year, "status": "PASS", "games": len(checks),
              "innings": sum(c["innings"] for c in checks), "pitch_rows": len(data),
              "game_checks": checks}
    (root / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "game_checks"}))


if __name__ == "__main__":
    main()

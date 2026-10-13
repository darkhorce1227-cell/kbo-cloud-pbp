#!/usr/bin/env python3
"""Audit pitch granularity in downloaded postseason raw relay vs exported PBP.

This is diagnostic, NOT an assertion that Naver's pitch events are complete.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from kbo_pbp import relay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--output", default="historical_ps")
    args = ap.parse_args()
    root = Path(args.output) / str(args.year)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(root / "postseason_pitches.parquet")
    rows = []
    for entry in manifest["games"]:
        if entry["status"] != "PASS":
            continue
        gid = str(entry["game_id"])
        raw_blocks = 0
        event_types = Counter()
        samples = []
        parse_failures = []
        for inning in range(1, int(entry["innings"]) + 1):
            with gzip.open(root / "raw" / f"{gid}_inning_{inning:02}.json.gz", "rt", encoding="utf-8") as f:
                payload = json.load(f)
            blocks = payload.get("textRelays", [])
            raw_blocks += len(blocks)
            try:
                events = list(relay.events({inning: payload}))
                for event in events:
                    event_types[str(event.get("type"))] += 1
                    if len(samples) < 3:
                        samples.append({"type": str(event.get("type")), "text": str(event.get("text", ""))[:90]})
            except Exception as exc:
                parse_failures.append(f"inning {inning}: {type(exc).__name__}: {exc}")
        selected = data[data["game_pk"].astype(str) == gid]
        pitch_rows = int((selected["pitch_number"] > 0).sum())
        rows.append({"year": args.year, "game_id": gid, "innings": entry["innings"],
                     "raw_blocks": raw_blocks, "parsed_events": sum(event_types.values()),
                     "event_types": json.dumps(dict(event_types), ensure_ascii=False),
                     "exported_rows": len(selected), "exported_pitch_rows": pitch_rows,
                     "pitch_rows_per_inning": round(pitch_rows / int(entry["innings"]), 2),
                     "samples": json.dumps(samples, ensure_ascii=False),
                     "parse_failures": json.dumps(parse_failures, ensure_ascii=False)})
    report = pd.DataFrame(rows)
    if report.empty:
        raise RuntimeError("No completed games to audit")
    report.to_csv(root / "pitch_integrity_audit.csv", index=False)
    summary = {"year": args.year, "games": len(report),
               "exported_pitch_rows": int(report["exported_pitch_rows"].sum()),
               "pitch_rows_per_game": round(report["exported_pitch_rows"].mean(), 1),
               "games_under_100_pitches": int((report["exported_pitch_rows"] < 100).sum()),
               "games_with_parse_errors": int(report["parse_failures"].ne("[]").sum())}
    (root / "pitch_integrity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect pre-2012 relay pitch gaps without modifying any dataset."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from kbo_pbp import relay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=[2009, 2010, 2011, 2012])
    p.add_argument("--root", default="historical_ps")
    args = p.parse_args()
    root = Path(args.root)
    all_rows = []
    examples = {}
    for year in args.years:
        folder = root / str(year)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        frame = pd.read_parquet(folder / "postseason_pitches.parquet")
        season_examples = {}
        for item in manifest["games"]:
            if item["status"] != "PASS":
                continue
            gid = str(item["game_id"])
            game = {}
            raw_types = Counter()
            raw_pitches = 0
            for inning in range(1, int(item["innings"]) + 1):
                with gzip.open(folder / "raw" / f"{gid}_inning_{inning:02}.json.gz", "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                game[inning] = payload
                for block in payload["textRelays"]:
                    for opt in block.get("textOptions", []):
                        raw_types[str(opt.get("type"))] += 1
                        raw_pitches += str(opt.get("type")) == str(relay.PITCH)
                        typ = str(opt.get("type"))
                        if len(season_examples.get(typ, [])) < 3:
                            season_examples.setdefault(typ, []).append({
                                "game_id": gid, "inning": inning,
                                "text": str(opt.get("text", ""))[:180],
                                "pitchNum": opt.get("pitchNum"),
                                "seqno": opt.get("seqno"),
                                "pitchResult": opt.get("pitchResult"),
                                "keys": sorted(opt.keys())})
            parsed = relay.events(game)
            parsed_pitches = [e for e in parsed if str(e.get("type")) == str(relay.PITCH)]
            selected = frame[frame["game_pk"].astype(str) == gid]
            all_rows.append({"year": year, "game_id": gid,
                             "raw_pitch_events": raw_pitches,
                             "parsed_pitch_events": len(parsed_pitches),
                             "exported_rows": len(selected),
                             "raw_minus_parsed": raw_pitches - len(parsed_pitches),
                             "parsed_minus_exported": len(parsed_pitches) - len(selected),
                             "raw_type_counts": json.dumps(dict(raw_types), ensure_ascii=False),
                             "parsed_pitch_numbers_zero": sum(not e.get("pitchNum") for e in parsed_pitches)})
        examples[str(year)] = season_examples
    df = pd.DataFrame(all_rows)
    df.to_csv(root / "raw_pitch_gap_by_game.csv", index=False)
    (root / "raw_pitch_gap_samples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = df.groupby("year")[["raw_pitch_events", "parsed_pitch_events", "exported_rows",
                                   "raw_minus_parsed", "parsed_minus_exported"]].sum()
    print(summary.to_string(), flush=True)
    print("NOTE: raw counts can contain retransmissions; this is diagnostic only.", flush=True)


if __name__ == "__main__":
    main()

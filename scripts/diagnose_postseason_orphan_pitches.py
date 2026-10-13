#!/usr/bin/env python3
"""Identify pitches outside an open plate appearance in sparse old relays."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from kbo_pbp import relay


def audit(game):
    pa_open = False
    outcome_seen = False
    half = None
    previous = []
    orphan = []
    all_pitches = 0
    counts = Counter()
    for event in relay.events(game):
        position = (event["inning"], event["half"])
        typ = event["type"]
        if position != half:
            pa_open, outcome_seen = False, False
            half = position
        if typ == relay.BATTER:
            if not pa_open or outcome_seen:
                pa_open, outcome_seen = True, False
        if typ == relay.PITCH:
            all_pitches += 1
            if not pa_open:
                counts["NO_BATTER_PA"] += 1
                if len(orphan) < 12:
                    orphan.append({"inning": event["inning"], "half": event["half"],
                                   "pitchNum": event.get("pitchNum"), "text": event.get("text"),
                                   "state": event.get("currentGameState"),
                                   "previous_events": previous[-4:]})
        if typ in relay.OUTCOMES and pa_open:
            outcome_seen = True
        previous.append({"type": typ, "text": str(event.get("text", ""))[:80],
                         "inning": event["inning"], "half": event["half"]})
        if len(previous) > 6:
            previous.pop(0)
    return all_pitches, counts, orphan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=[2009, 2010, 2011, 2012])
    args = p.parse_args()
    root = Path("historical_ps")
    rows, samples = [], {}
    for year in args.years:
        folder = root / str(year)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        frame = pd.read_parquet(folder / "postseason_pitches.parquet")
        for entry in manifest["games"]:
            if entry["status"] != "PASS":
                continue
            gid = str(entry["game_id"])
            game = {}
            for inning in range(1, int(entry["innings"]) + 1):
                with gzip.open(folder / "raw" / f"{gid}_inning_{inning:02}.json.gz", "rt", encoding="utf-8") as f:
                    game[inning] = json.load(f)
            n, counts, examples = audit(game)
            exported = len(frame[frame["game_pk"].astype(str) == gid])
            rows.append({"year": year, "game_id": gid, "parsed_pitches": n,
                         "exported_rows": exported, "gap": n-exported,
                         "pitches_before_batter": counts["NO_BATTER_PA"]})
            if examples:
                samples[gid] = examples
    df = pd.DataFrame(rows)
    df.to_csv(root / "orphan_pitch_audit.csv", index=False)
    (root / "orphan_pitch_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.groupby("year")[["parsed_pitches", "exported_rows", "gap",
                               "pitches_before_batter"]].sum().to_string(), flush=True)
    print("Orphan samples by game:", {k: len(v) for k,v in samples.items()}, flush=True)


if __name__ == "__main__":
    main()

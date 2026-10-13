#!/usr/bin/env python3
"""Automatic conservative quality gate for postseason pitch-level datasets."""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from kbo_pbp import relay


def classify(coverage_status, gaps, sparse_games):
    if coverage_status != "COMPLETE":
        return "PARTIAL", "RECOLLECT_OR_DOCUMENT_MISSING_GAMES"
    if gaps > 0 and sparse_games > 0:
        return "EXPORT_GAP_AND_SPARSE_SOURCE", "DEBUG_EXPORTER_AND_TREAT_SOURCE_AS_INCOMPLETE"
    if gaps > 0:
        return "EXPORT_GAP", "DEBUG_EXPORTER"
    if sparse_games > 0:
        return "SPARSE_SOURCE", "DO_NOT_USE_AS_FULL_PITCH_PBP"
    return "SCREEN_PASS", "ELIGIBLE_FOR_SCREENED_MASTER_DATASET"


def evaluate(root, year):
    folder = root / str(year)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(folder / "postseason_pitches.parquet")
    rows = []
    for entry in manifest["games"]:
        if entry["status"] != "PASS":
            continue
        gid = str(entry["game_id"])
        game = {}
        for inning in range(1, int(entry["innings"]) + 1):
            with gzip.open(folder / "raw" / f"{gid}_inning_{inning:02}.json.gz", "rt", encoding="utf-8") as f:
                game[inning] = json.load(f)
        events = relay.events(game)
        pitches = sum(e.get("type") == relay.PITCH for e in events)
        exported = len(df[df["game_pk"].astype(str) == gid])
        rows.append({
            "year": year,
            "game_id": gid,
            "innings": entry["innings"],
            "parsed_pitches": pitches,
            "exported_rows": exported,
            "pitch_gap": pitches - exported,
            "pitches_per_inning": round(pitches / int(entry["innings"]), 1),
        })
    checks = pd.DataFrame(rows)
    if checks.empty:
        raise RuntimeError("No games available for quality audit")
    checks.to_csv(folder / "quality_by_game.csv", index=False)
    gaps = int(checks["pitch_gap"].clip(lower=0).sum())
    sparse = int((checks["pitches_per_inning"] < 15).sum())
    quality, next_action = classify(validation["status"], gaps, sparse)
    return {
        "year": year,
        "quality": quality,
        "next_action": next_action,
        "coverage_status": validation["status"],
        "scheduled_games": validation["scheduled_games"],
        "exported_games": validation["exported_games"],
        "pitch_rows": validation["pitch_rows"],
        "unexported_pitch_events": gaps,
        "games_below_15_pitches_per_inning": sparse,
        "note": "15 pitches/inning is a screening threshold, not official completeness proof.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--root", default="historical_ps")
    args = p.parse_args()
    root = Path(args.root)
    report = []
    for year in args.years:
        try:
            item = evaluate(root, year)
        except Exception as exc:
            item = {
                "year": year,
                "quality": "ERROR",
                "next_action": "INSPECT_PIPELINE_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        report.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (root / "quality_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report).to_csv(root / "quality_summary.csv", index=False)
    if any(x["quality"] == "ERROR" for x in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

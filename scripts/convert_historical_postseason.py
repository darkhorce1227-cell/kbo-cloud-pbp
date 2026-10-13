#!/usr/bin/env python3
"""Validate the downloaded postseason relay archive and export pitch-level PBP."""
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
    folder = Path(args.output) / str(args.year)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or manifest["errors"] or not manifest["games"]:
        raise RuntimeError("Collection manifest is incomplete")
    schedule = pd.read_csv(folder / "schedule.csv", dtype={"gameId": str})
    if len(schedule) != len(manifest["games"]):
        raise RuntimeError("Schedule and manifest game counts differ")
    if schedule["gameId"].duplicated().any():
        raise RuntimeError("Duplicate game IDs in schedule")
    lookup = schedule.set_index("gameId", drop=False)
    frames = []
    details = []
    for item in manifest["games"]:
        gid = item["game_id"]
        count = int(item["innings"])
        if item["status"] != "PASS" or int(item["fetched"]) != count:
            raise RuntimeError(f"Incomplete game: {gid}")
        if gid not in lookup.index:
            raise RuntimeError(f"Game absent from schedule: {gid}")
        game = {}
        for inning in range(1, count + 1):
            path = folder / "raw" / f"{gid}_inning_{inning:02}.json.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with gzip.open(path, "rt", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict) or not isinstance(payload.get("textRelays"), list):
                raise RuntimeError(f"Invalid relay payload: {path}")
            game[inning] = payload
        meta = statcast.meta_from_schedule(lookup.loc[gid])
        frame = statcast.game_frame(game, meta)
        if frame.empty:
            raise RuntimeError(f"Zero pitch rows for game: {gid}")
        frames.append(frame)
        details.append({"game_id": gid, "innings": count, "pitch_rows": len(frame)})
    result = pd.concat(frames, ignore_index=True)
    if result["game_pk"].nunique() != len(manifest["games"]):
        raise RuntimeError("Converted game count differs from manifest")
    csv = folder / f"kbo_ps_{args.year}_pbp.csv.gz"
    parquet = folder / f"kbo_ps_{args.year}_pbp.parquet"
    result.to_csv(csv, index=False, compression="gzip")
    result.to_parquet(parquet, index=False)
    summary = {
        "year": args.year,
        "status": "PASS",
        "games": len(details),
        "innings": sum(d["innings"] for d in details),
        "pitch_rows": len(result),
        "game_details": details,
        "note": "Validates relay presence and parseability; independent official-score audit not yet performed.",
    }
    (folder / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "game_details"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

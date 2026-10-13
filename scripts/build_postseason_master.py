#!/usr/bin/env python3
"""Combine screened historical postseason pitch datasets into master Parquet files."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="historical_ps")
    args = p.parse_args()
    root = Path(args.root)

    summary_path = root / "quality_summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    all_frames = []
    screened_frames = []
    index_rows = []

    for item in report:
        year = int(item["year"])
        quality = item.get("quality", "ERROR")
        folder = root / str(year)
        path = folder / "postseason_pitches.parquet"
        if not path.exists():
            index_rows.append({
                "year": year,
                "quality": quality,
                "rows": 0,
                "included_all_available": False,
                "included_screen_pass": False,
                "reason": "postseason_pitches.parquet missing",
            })
            continue

        frame = pd.read_parquet(path).copy()
        frame.insert(0, "season", year)
        frame.insert(1, "data_quality", quality)
        frame.insert(2, "coverage_status", item.get("coverage_status"))
        all_frames.append(frame)
        is_screen_pass = quality == "SCREEN_PASS"
        if is_screen_pass:
            screened_frames.append(frame)
        index_rows.append({
            "year": year,
            "quality": quality,
            "rows": len(frame),
            "included_all_available": True,
            "included_screen_pass": is_screen_pass,
            "coverage_status": item.get("coverage_status"),
            "unexported_pitch_events": item.get("unexported_pitch_events"),
            "games_below_15_pitches_per_inning": item.get("games_below_15_pitches_per_inning"),
            "next_action": item.get("next_action"),
        })

    if not all_frames:
        raise RuntimeError("No postseason Parquet files were available to combine")

    all_data = pd.concat(all_frames, ignore_index=True, sort=False)
    screened = (pd.concat(screened_frames, ignore_index=True, sort=False)
                if screened_frames else all_data.iloc[0:0].copy())

    all_data.to_parquet(root / "postseason_all_available.parquet", index=False)
    screened.to_parquet(root / "postseason_screen_pass.parquet", index=False)
    pd.DataFrame(index_rows).sort_values("year").to_csv(
        root / "postseason_dataset_index.csv", index=False
    )

    result = {
        "available_years": sorted({int(x["season"]) for _, x in all_data[["season"]].iterrows()}),
        "screen_pass_years": sorted({int(x["season"]) for _, x in screened[["season"]].iterrows()}),
        "all_available_rows": len(all_data),
        "screen_pass_rows": len(screened),
        "warning": "SCREEN_PASS is an automated screening result, not independent official-stat completeness verification.",
    }
    (root / "postseason_master_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

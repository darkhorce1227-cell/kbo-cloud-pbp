#!/usr/bin/env python3
"""Combine screened historical postseason pitch datasets into master Parquet files."""
import argparse
import json
from pathlib import Path

import pandas as pd


def put_front(frame, values):
    for key, value in values.items():
        frame[key] = value
    front = list(values)
    rest = [c for c in frame.columns if c not in front]
    return frame[front + rest]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="historical_ps")
    args = p.parse_args()
    root = Path(args.root)

    report = json.loads((root / "quality_summary.json").read_text(encoding="utf-8"))
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
        frame = put_front(frame, {
            "season": year,
            "data_quality": quality,
            "coverage_status": item.get("coverage_status"),
        })
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
            "games_under_100_raw_pitch_events": item.get("games_under_100_raw_pitch_events"),
            "next_action": item.get("next_action"),
        })

    if not all_frames:
        raise RuntimeError("No postseason Parquet files were available to combine")

    all_data = pd.concat(all_frames, ignore_index=True, sort=False)
    screened = (
        pd.concat(screened_frames, ignore_index=True, sort=False)
        if screened_frames else all_data.iloc[0:0].copy()
    )

    all_data.to_parquet(root / "postseason_all_available.parquet", index=False)
    screened.to_parquet(root / "postseason_screen_pass.parquet", index=False)
    pd.DataFrame(index_rows).sort_values("year").to_csv(
        root / "postseason_dataset_index.csv", index=False
    )

    unavailable = []
    unavailable_path = root / "unavailable_years.txt"
    if unavailable_path.exists():
        unavailable = [int(x) for x in unavailable_path.read_text().split() if x.strip()]

    result = {
        "available_years": sorted(all_data["season"].dropna().astype(int).unique().tolist()),
        "screen_pass_years": sorted(screened["season"].dropna().astype(int).unique().tolist()),
        "unavailable_years": sorted(unavailable),
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

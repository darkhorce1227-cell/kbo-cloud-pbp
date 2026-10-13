#!/usr/bin/env python3
"""Compare raw Naver relay event types and pitch recognition across seasons."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from kbo_pbp import relay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2008, 2009, 2010, 2011, 2012, 2021])
    ap.add_argument("--output", default="historical_ps")
    args = ap.parse_args()
    root = Path(args.output)
    report = []
    for year in args.years:
        base = root / str(year)
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        season_types, sample_by_type = Counter(), {}
        game_rows = []
        for entry in manifest["games"]:
            if entry["status"] != "PASS":
                continue
            gid = str(entry["game_id"])
            game = {}
            for inning in range(1, int(entry["innings"]) + 1):
                with gzip.open(base / "raw" / f"{gid}_inning_{inning:02}.json.gz", "rt", encoding="utf-8") as f:
                    game[inning] = json.load(f)
            events = relay.events(game)
            types = Counter(str(e.get("type")) for e in events)
            season_types.update(types)
            for e in events:
                typ = str(e.get("type"))
                if typ not in sample_by_type:
                    sample_by_type[typ] = {"type": typ, "text": str(e.get("text", ""))[:140],
                                           "pitchNum": e.get("pitchNum"),
                                           "pitchResult": e.get("pitchResult")}
            game_rows.append({"year": year, "game_id": gid, "events": len(events),
                              "pitch_events": types.get(str(relay.PITCH), 0),
                              "event_types": json.dumps(dict(types), ensure_ascii=False)})
        pd.DataFrame(game_rows).to_csv(base / "relay_event_comparison.csv", index=False)
        item = {"year": year, "games": len(game_rows),
                "event_types": dict(season_types),
                "recognized_pitch_events": season_types.get(str(relay.PITCH), 0),
                "samples_by_type": sample_by_type}
        report.append(item)
        print(json.dumps({"year": year, "games": item["games"],
                          "recognized_pitch_events": item["recognized_pitch_events"],
                          "event_types": item["event_types"]}, ensure_ascii=False), flush=True)
    (root / "relay_event_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

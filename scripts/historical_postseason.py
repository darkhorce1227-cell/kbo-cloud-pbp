#!/usr/bin/env python3
"""Download historical KBO postseason pitch-by-pitch data from Naver.

Uses the upstream parser already installed by requirements.txt.
Never marks incomplete seasons as successful.
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import pandas as pd
from kbo_pbp import naver, schedule

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--output", default="historical_ps")
    ap.add_argument("--pause", type=float, default=0.25)
    args = ap.parse_args()
    if not 2000 <= args.year <= 2026:
        ap.error("year must be between 2000 and 2026")
    out = Path(args.output) / str(args.year)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"year": args.year, "status": "INCOMPLETE", "games": [], "errors": []}
    try:
        games = schedule.playable(schedule.fetch(args.year), rounds=schedule.POSTSEASON)
        if games.empty:
            raise RuntimeError("No postseason games in schedule API (not proof none existed)")
        games.to_csv(out / "schedule.csv", index=False)
        for _, game in games.iterrows():
            gid = str(game.get("gameId") or game.get("id") or "")
            innings = int(game["final_inning"])
            item = {"game_id": gid, "innings": innings, "fetched": 0, "status": "INCOMPLETE"}
            manifest["games"].append(item)
            if not gid:
                manifest["errors"].append("Missing gameId in schedule row")
                continue
            for inning in range(1, innings + 1):
                path = raw / f"{gid}_inning_{inning:02}.json.gz"
                try:
                    if not path.exists():
                        payload = naver.fetch_relay(gid, inning)
                        with gzip.open(path, "wt", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False)
                        time.sleep(args.pause)
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        json.load(f)
                    item["fetched"] += 1
                except Exception as exc:
                    manifest["errors"].append(f"{gid} inning {inning}: {type(exc).__name__}: {exc}")
            item["status"] = "PASS" if item["fetched"] == innings else "INCOMPLETE"
    except Exception as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
    manifest["status"] = "PASS" if manifest["games"] and not manifest["errors"] and all(g["status"] == "PASS" for g in manifest["games"]) else "INCOMPLETE"
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"year": args.year, "status": manifest["status"], "games": len(manifest["games"]), "errors": manifest["errors"][:5]}, ensure_ascii=False))
    if manifest["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()

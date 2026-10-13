#!/usr/bin/env python3
"""Download historical KBO postseason pitch-by-pitch data from Naver.

Older schedule payloads can report final_inning=1 even for completed full games.
For those seasons we probe inning-by-inning until the first empty inning after 9.
"""
import argparse
import gzip
import json
import time
from pathlib import Path

from kbo_pbp import naver, schedule

MAX_PROBE_INNING = 18


def load_or_fetch(path, gid, inning, pause):
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    payload = naver.fetch_relay(gid, inning)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    time.sleep(pause)
    return payload


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
    manifest = {"year": args.year, "status": "FAILED", "games": [], "errors": []}

    try:
        games = schedule.playable(schedule.fetch(args.year), rounds=schedule.POSTSEASON)
        if games.empty:
            raise RuntimeError("No postseason games in schedule API (not proof none existed)")
        games.to_csv(out / "schedule.csv", index=False)

        for _, game in games.iterrows():
            gid = str(game.get("gameId") or game.get("id") or "")
            scheduled_final = int(game.get("final_inning") or 0)
            # final_inning=1 is a known legacy schedule defect. Treat a normal
            # postseason game as requiring at least innings 1..9, then probe
            # extras until the first empty/missing inning.
            required_through = scheduled_final if scheduled_final > 1 else 9
            item = {
                "game_id": gid,
                "scheduled_final_inning": scheduled_final,
                "innings": 0,
                "fetched": 0,
                "status": "INCOMPLETE",
            }
            manifest["games"].append(item)
            if not gid:
                manifest["errors"].append("Missing gameId in schedule row")
                continue

            fetched_innings = []
            for inning in range(1, MAX_PROBE_INNING + 1):
                path = raw / f"{gid}_inning_{inning:02}.json.gz"
                required = inning <= required_through
                try:
                    payload = load_or_fetch(path, gid, inning, args.pause)
                    blocks = payload.get("textRelays") or []
                    if not blocks:
                        if required:
                            manifest["errors"].append(
                                f"{gid} inning {inning}: empty textRelays before required boundary"
                            )
                        else:
                            break
                    else:
                        fetched_innings.append(inning)
                except Exception as exc:
                    if required:
                        manifest["errors"].append(
                            f"{gid} inning {inning}: {type(exc).__name__}: {exc}"
                        )
                    else:
                        break

            item["fetched"] = len(fetched_innings)
            item["innings"] = max(fetched_innings) if fetched_innings else required_through
            expected = set(range(1, item["innings"] + 1))
            item["missing_innings"] = sorted(expected - set(fetched_innings))
            item["status"] = "PASS" if not item["missing_innings"] and item["innings"] >= required_through else "INCOMPLETE"

    except Exception as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")

    passed = sum(g["status"] == "PASS" for g in manifest["games"])
    manifest["status"] = (
        "COMPLETE" if passed == len(manifest["games"]) and passed > 0 and not manifest["errors"]
        else "PARTIAL" if passed > 0
        else "FAILED"
    )
    manifest["complete_games"] = passed
    manifest["missing_games"] = [g["game_id"] for g in manifest["games"] if g["status"] != "PASS"]
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "year": args.year,
        "status": manifest["status"],
        "games": len(manifest["games"]),
        "errors": manifest["errors"][:5],
    }, ensure_ascii=False))
    if manifest["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

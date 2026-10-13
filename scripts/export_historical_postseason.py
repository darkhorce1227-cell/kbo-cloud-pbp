#!/usr/bin/env python3
"""Export complete games only, with explicit season coverage and missing innings."""
import argparse
import gzip
import json
from pathlib import Path
import pandas as pd
from kbo_pbp import statcast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--output", default="historical_ps")
    args = ap.parse_args()
    root = Path(args.output) / str(args.year)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] == "FAILED" or not manifest["games"]:
        raise RuntimeError("No complete games available")
    schedule = pd.read_csv(root / "schedule.csv", dtype={"gameId": str}).set_index("gameId", drop=False)
    if len(schedule) != len(manifest["games"]) or schedule.index.has_duplicates:
        raise RuntimeError("Schedule/manifest mismatch")
    frames, checks, missing, parse_errors = [], [], [], []
    for entry in manifest["games"]:
        gid, innings = str(entry["game_id"]), int(entry["innings"])
        if gid not in schedule.index:
            raise RuntimeError(f"Game absent from schedule: {gid}")
        if entry["status"] != "PASS" or entry["fetched"] != innings:
            for inning in range(1, innings + 1):
                path = root / "raw" / f"{gid}_inning_{inning:02}.json.gz"
                if not path.exists():
                    missing.append({"game_id": gid, "inning": inning, "reason": "FETCH_MISSING"})
                else:
                    try:
                        with gzip.open(path, "rt", encoding="utf-8") as f:
                            payload = json.load(f)
                        if not isinstance(payload, dict) or not isinstance(payload.get("textRelays"), list):
                            raise ValueError("Unexpected relay structure")
                    except Exception:
                        missing.append({"game_id": gid, "inning": inning, "reason": "INVALID_RELAY"})
            continue
        try:
            game = {}
            for inning in range(1, innings + 1):
                with gzip.open(root / "raw" / f"{gid}_inning_{inning:02}.json.gz",
                               "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                relay = payload.get("result", {}).get("textRelayData", payload)
                if not isinstance(relay, dict) or not isinstance(relay.get("textRelays"), list):
                    raise ValueError(f"Unexpected relay structure in inning {inning}")
                game[inning] = relay
            frame = statcast.game_frame(game, statcast.meta_from_schedule(schedule.loc[gid]))
            if frame.empty:
                raise ValueError("No pitch rows")
            frame["season"] = args.year
            frame["round_code"] = schedule.loc[gid].get("roundCode")
            frames.append(frame)
            checks.append({"game_id": gid, "innings": innings, "rows": len(frame),
                           "unique_pa": int(frame["at_bat_number"].nunique())})
        except Exception as exc:
            parse_errors.append({"game_id": gid, "reason": f"{type(exc).__name__}: {exc}"})
    if not frames:
        raise RuntimeError("No games could be parsed")
    data = pd.concat(frames, ignore_index=True)
    if data[["game_pk", "inning", "at_bat_number", "pitch_number"]].isna().any().any():
        raise RuntimeError("Missing essential pitch keys")
    total = len(manifest["games"])
    status = "COMPLETE" if len(checks) == total and not manifest["errors"] and not parse_errors else "PARTIAL"
    report = {"year": args.year, "status": status, "scheduled_games": total,
              "exported_games": len(checks), "coverage": len(checks) / total,
              "pitch_rows": len(data), "game_checks": checks,
              "missing_games": sorted(set(g["game_id"] for g in manifest["games"]) -
                                      set(g["game_id"] for g in checks)),
              "missing_innings": missing, "parse_errors": parse_errors,
              "collection_errors": manifest["errors"]}
    data.to_csv(root / "postseason_pitches.csv.gz", index=False, compression="gzip")
    data.to_parquet(root / "postseason_pitches.parquet", index=False)
    pd.DataFrame(missing, columns=["game_id", "inning", "reason"]).to_csv(
        root / "missing_innings.csv", index=False)
    (root / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("game_checks", "collection_errors", "missing_innings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

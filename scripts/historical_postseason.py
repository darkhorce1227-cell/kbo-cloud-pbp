#!/usr/bin/env python3
"""Download historical KBO postseason pitch-by-pitch data from Naver.

Older schedule payloads can report final_inning=1 even for completed full games.
For those seasons we probe inning-by-inning until the first empty inning after 9.
Some legacy schedule rows also expose a short gameId even when the relay endpoint
is stored under a year-suffixed or modern date-based gameId, so those variants
are probed automatically before a game is declared missing.
"""
import argparse
import gzip
import json
import time
from pathlib import Path

from kbo_pbp import naver, schedule

MAX_PROBE_INNING = 18


def _unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def candidate_game_ids(game, year, gid):
    """Return conservative alternate IDs for the same scheduled game."""
    date = str(game.get("gameDate") or "")[:10].replace("-", "")
    away = str(game.get("awayTeamCode") or "").strip()
    home = str(game.get("homeTeamCode") or "").strip()
    year_s = str(year)

    candidates = [gid]
    # Several surviving 2011/2013 postseason relays use the legacy short ID
    # with the season appended, while adjacent schedule rows expose only the
    # short form. Try that exact reversible transformation first.
    if gid and not gid.endswith(year_s):
        candidates.append(f"{gid}{year_s}")

    # Current Naver game IDs use YYYYMMDD + away + home + header + YYYY.
    # Try only the same date/teams and plausible header values.
    if len(date) == 8 and away and home:
        for header in ("0", "1", "2"):
            candidates.append(f"{date}{away}{home}{header}{year_s}")
        candidates.append(f"{date}{away}{home}0")

    return _unique(candidates)


def resolve_relay_id(game, year, gid, pause):
    """Find an ID whose inning-1 relay exists and has text blocks."""
    tried, errors = [], []
    for candidate in candidate_game_ids(game, year, gid):
        tried.append(candidate)
        try:
            payload = naver.fetch_relay(candidate, 1)
            time.sleep(pause)
            if payload.get("textRelays"):
                return candidate, payload, tried, errors
            errors.append(f"{candidate}: empty textRelays")
        except Exception as exc:
            time.sleep(pause)
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    return None, None, tried, errors


def load_or_fetch(path, relay_gid, inning, pause, prefetched=None):
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    payload = prefetched if prefetched is not None else naver.fetch_relay(relay_gid, inning)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    if prefetched is None:
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
                "relay_game_id": None,
                "id_recovered": False,
                "candidate_ids_tried": [],
                "scheduled_final_inning": scheduled_final,
                "innings": 0,
                "fetched": 0,
                "status": "INCOMPLETE",
            }
            manifest["games"].append(item)
            if not gid:
                manifest["errors"].append("Missing gameId in schedule row")
                continue

            relay_gid, first_payload, tried, candidate_errors = resolve_relay_id(
                game, args.year, gid, args.pause
            )
            item["candidate_ids_tried"] = tried
            item["candidate_probe_errors"] = candidate_errors
            if relay_gid is None:
                manifest["errors"].append(
                    f"{gid}: no relay candidate found; tried={','.join(tried)}"
                )
                item["innings"] = required_through
                item["missing_innings"] = list(range(1, required_through + 1))
                continue

            item["relay_game_id"] = relay_gid
            item["id_recovered"] = relay_gid != gid
            fetched_innings = []
            for inning in range(1, MAX_PROBE_INNING + 1):
                path = raw / f"{gid}_inning_{inning:02}.json.gz"
                required = inning <= required_through
                try:
                    payload = load_or_fetch(
                        path,
                        relay_gid,
                        inning,
                        args.pause,
                        prefetched=first_payload if inning == 1 else None,
                    )
                    blocks = payload.get("textRelays") or []
                    if not blocks:
                        if required:
                            manifest["errors"].append(
                                f"{gid} via {relay_gid} inning {inning}: empty textRelays before required boundary"
                            )
                        else:
                            break
                    else:
                        fetched_innings.append(inning)
                except Exception as exc:
                    if required:
                        manifest["errors"].append(
                            f"{gid} via {relay_gid} inning {inning}: {type(exc).__name__}: {exc}"
                        )
                    else:
                        break

            item["fetched"] = len(fetched_innings)
            item["innings"] = max(fetched_innings) if fetched_innings else required_through
            expected = set(range(1, item["innings"] + 1))
            item["missing_innings"] = sorted(expected - set(fetched_innings))
            item["status"] = (
                "PASS"
                if not item["missing_innings"] and item["innings"] >= required_through
                else "INCOMPLETE"
            )

    except Exception as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")

    passed = sum(g["status"] == "PASS" for g in manifest["games"])
    recovered = [
        {"game_id": g["game_id"], "relay_game_id": g.get("relay_game_id")}
        for g in manifest["games"]
        if g.get("id_recovered") and g["status"] == "PASS"
    ]
    manifest["status"] = (
        "COMPLETE" if passed == len(manifest["games"]) and passed > 0 and not manifest["errors"]
        else "PARTIAL" if passed > 0
        else "FAILED"
    )
    manifest["complete_games"] = passed
    manifest["recovered_game_ids"] = recovered
    manifest["missing_games"] = [g["game_id"] for g in manifest["games"] if g["status"] != "PASS"]
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "year": args.year,
        "status": manifest["status"],
        "games": len(manifest["games"]),
        "recovered_game_ids": recovered,
        "errors": manifest["errors"][:5],
    }, ensure_ascii=False))
    if manifest["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose historical Naver postseason relay availability without guessing.

Report schedule IDs and API response structure for every postseason game,
and compare a few known-good years. Does not alter or publish datasets.
"""
import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

import requests
from kbo_pbp import naver, schedule


def probe(game_id, inning):
    path = f"/schedule/games/{game_id}/relay"
    try:
        response = naver._session.get(naver.BASE + path, params={"inning": inning}, timeout=20)
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            return {"http_status": status, "category": "NON_JSON",
                    "body_preview": response.text[:180]}
        result = body.get("result") if isinstance(body, dict) else None
        relay = result.get("textRelayData") if isinstance(result, dict) else None
        if isinstance(relay, dict):
            blocks = relay.get("textRelays")
            return {"http_status": status, "category": "RELAY_PRESENT",
                    "blocks": len(blocks) if isinstance(blocks, list) else None,
                    "relay_keys": list(relay)[:15]}
        return {"http_status": status, "category": "NO_TEXT_RELAY",
                "top_keys": list(body)[:12] if isinstance(body, dict) else [],
                "result_keys": list(result)[:20] if isinstance(result, dict) else [],
                "api_code": body.get("code") if isinstance(body, dict) else None,
                "api_message": str(body.get("message", ""))[:180] if isinstance(body, dict) else ""}
    except requests.RequestException as exc:
        return {"category": "NETWORK_ERROR", "error": str(exc)[:250]}
    except Exception as exc:
        return {"category": "UNEXPECTED_ERROR", "error": repr(exc)[:250]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2008, 2009, 2010, 2011, 2012, 2021])
    ap.add_argument("--output", default="historical_ps_diagnostics")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"years": {}, "notes": "HTTP/JSON structure only; no game completeness assertion"}
    for year in args.years:
        record = {"year": year, "games": [], "counts": {}, "schedule_error": None}
        try:
            games = schedule.playable(schedule.fetch(year), rounds=schedule.POSTSEASON)
            record["schedule_games"] = len(games)
            for _, row in games.iterrows():
                gid = str(row.get("gameId") or row.get("id") or "")
                innings = int(row["final_inning"])
                result = probe(gid, 1)
                record["games"].append({"game_id": gid, "innings": innings, **result})
            record["counts"] = dict(Counter(g["category"] for g in record["games"]))
        except Exception as exc:
            record["schedule_error"] = f"{type(exc).__name__}: {exc}"
        report["years"][str(year)] = record
        print(json.dumps({"year": year, "schedule_games": record.get("schedule_games"),
                          "counts": record["counts"], "error": record["schedule_error"]},
                         ensure_ascii=False), flush=True)
    (out / "api_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

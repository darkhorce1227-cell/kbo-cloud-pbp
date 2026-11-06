"""Resolve immutable pitcher handedness from official KBO profiles, never coordinates.

No roster or historical availability is inferred from a current profile. Cached
evidence records its retrieval date, URL, profile name, and source SHA256.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

import pandas as pd


def parse_profile(source: str) -> dict:
    def label(suffix: str) -> str:
        match = re.search(r'id="[^"]*playerProfile_' + suffix + r'"[^>]*>(.*?)</span>', source, re.S)
        if not match:
            return ""
        return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()

    position = label("lblPosition")
    # KBO writes right/left underhand as 우언/좌언, not 우투/좌투.
    match = re.search(r"([좌우양])(?:투|언)", position)
    hand = {"좌": "L", "우": "R", "양": "S"}.get(match.group(1), "UNKNOWN") if match else "UNKNOWN"
    return {"name": label("lblName"), "profile_position": position, "throws": hand}


def collect(ids: list[str], cache: Path, workers: int = 6) -> dict:
    cache.mkdir(parents=True, exist_ok=True)

    def one(player_id: str):
        target = cache / (player_id + ".json")
        if target.exists():
            old = json.loads(target.read_text())
            if old.get("profile_position"):
                match = re.search(r"([좌우양])(?:투|언)", old["profile_position"])
                old["throws"] = {"좌": "L", "우": "R", "양": "S"}.get(match.group(1), "UNKNOWN") if match else "UNKNOWN"
                old["status"] = "VERIFIED_PROFILE" if old["throws"] in {"L", "R"} and old.get("name") else "UNRESOLVED"
                target.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n")
                return player_id, old
            # Transient download failures may be retried on an explicit rerun.
        url = "https://www.koreabaseball.com/Record/Player/PitcherDetail/Basic.aspx?playerId=" + player_id
        record = {"player_id": player_id, "url": url,
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KBO-research/1.0"})
            with urllib.request.urlopen(req, timeout=18) as response:
                raw = response.read()
            record.update(parse_profile(raw.decode("utf-8-sig", errors="replace")))
            record["sha256_html"] = hashlib.sha256(raw).hexdigest()
            record["status"] = "VERIFIED_PROFILE" if record["throws"] in {"L", "R"} and record["name"] else "UNRESOLVED"
        except Exception as exc:
            record.update(status="UNRESOLVED", throws="UNKNOWN", error=type(exc).__name__)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return player_id, record

    result = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (player_id, record) in enumerate(pool.map(one, ids), start=1):
            result[player_id] = record
            if index % 40 == 0:
                print(json.dumps({"profiles_complete": index, "total": len(ids)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitchers", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.pitchers)
    ids = sorted({str(int(x)) for x in frame.pitcher.dropna()})
    result = collect(ids, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": "OFFICIAL_KBO_PROFILE_STATIC_HANDEDNESS_ONLY", "players": result}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"profiles": len(result), "verified": sum(r["status"] == "VERIFIED_PROFILE" for r in result.values()), "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

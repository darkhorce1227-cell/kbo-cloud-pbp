from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import kbo_pbp
from kbo_pbp import schedule

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_history"
YEARS = (2024, 2025)

REQUIRED = {
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "batter", "pitcher", "batter_name", "pitcher_name", "events",
}
HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
AB_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "strikeout",
    "fielders_choice", "double_play", "triple_play", "field_error",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_year(year: int) -> tuple[pd.DataFrame, dict[str, int]]:
    errors = kbo_pbp.download(year)
    if errors:
        raise RuntimeError(f"{year}: {len(errors)} download failures; sample={list(errors.items())[:5]}")

    df = kbo_pbp.season(year)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"kbo_pbp.season({year}) did not return DataFrame")

    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"{year}: missing columns {missing}")

    dupes = int(df.duplicated(["game_pk", "at_bat_number", "pitch_number"], keep=False).sum())
    if dupes:
        raise RuntimeError(f"{year}: duplicate stable pitch keys={dupes}")

    season_schedule = schedule.fetch(year)
    playable = schedule.playable(season_schedule)
    expected_ids = set(playable["gameId"].astype(str))
    found_ids = set(df["game_pk"].astype(str).unique())
    missing_games = sorted(expected_ids - found_ids)
    unexpected_games = sorted(found_ids - expected_ids)
    if missing_games or unexpected_games:
        raise RuntimeError(
            f"{year}: schedule/PBP mismatch expected={len(expected_ids)} found={len(found_ids)} "
            f"missing={missing_games[:5]} unexpected={unexpected_games[:5]}"
        )

    audit = {
        "expected_games": len(expected_ids),
        "found_games": len(found_ids),
        "duplicate_stable_keys": dupes,
        "download_failures": 0,
    }
    return df, audit


def final_pa(df: pd.DataFrame, year: int) -> pd.DataFrame:
    x = df.copy()
    x["_event"] = x["events"].astype("string")
    # Event labels repeat on pitch rows. Keep the last pitch row per PA only.
    x = x.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    pa = x.groupby(["game_pk", "at_bat_number"], as_index=False).tail(1).copy()
    pa = pa[pa["_event"].notna()].copy()
    pa["season"] = year
    return pa


def aggregate_pvb(pa: pd.DataFrame) -> pd.DataFrame:
    e = pa["_event"].fillna("")
    pa["PA"] = 1
    pa["AB"] = e.isin(AB_EVENTS).astype(int)
    pa["H"] = e.isin(HITS).astype(int)
    pa["2B"] = e.eq("double").astype(int)
    pa["3B"] = e.eq("triple").astype(int)
    pa["HR"] = e.eq("home_run").astype(int)
    pa["XBH"] = e.isin({"double", "triple", "home_run"}).astype(int)
    pa["BB"] = e.eq("walk").astype(int)
    pa["HBP"] = e.eq("hit_by_pitch").astype(int)
    pa["K"] = e.eq("strikeout").astype(int)
    pa["SF"] = e.eq("sac_fly").astype(int)
    pa["TB"] = e.map(HITS).fillna(0).astype(int)

    keys = ["season", "batter", "batter_name", "pitcher", "pitcher_name"]
    cols = ["PA", "AB", "H", "2B", "3B", "HR", "XBH", "BB", "HBP", "K", "SF", "TB"]
    out = pa.groupby(keys, dropna=False)[cols].sum().reset_index()

    out["AVG"] = out["H"].div(out["AB"].where(out["AB"] > 0))
    obp_den = out["AB"] + out["BB"] + out["HBP"] + out["SF"]
    out["OBP"] = (out["H"] + out["BB"] + out["HBP"]).div(obp_den.where(obp_den > 0))
    out["SLG"] = out["TB"].div(out["AB"].where(out["AB"] > 0))
    out["OPS"] = out["OBP"] + out["SLG"]
    out["SMALL_SAMPLE_MULTI_XBH"] = out["XBH"].ge(2)

    rate_cols = ["AVG", "OBP", "SLG", "OPS"]
    out[rate_cols] = out[rate_cols].round(4)
    return out.sort_values(["season", "batter_name", "pitcher_name", "batter", "pitcher"]).reset_index(drop=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    diagnostics = {}

    for year in YEARS:
        df, audit = load_year(year)
        pa = final_pa(df, year)
        frames.append(aggregate_pvb(pa))
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        diagnostics[str(year)] = {
            **audit,
            "pitch_rows": int(len(df)),
            "columns": int(len(df.columns)),
            "final_pa_rows": int(len(pa)),
            "min_game_date": str(dates.min().date()),
            "max_game_date": str(dates.max().date()),
        }

    pvb = pd.concat(frames, ignore_index=True)
    if pvb.duplicated(["season", "batter", "pitcher"]).any():
        raise RuntimeError("duplicate season/batter/pitcher PvB summary keys")

    csv_plain = OUT / "kbo_pvb_2024_2025.csv"
    csv_gz = OUT / "kbo_pvb_2024_2025.csv.gz"
    parquet = OUT / "kbo_pvb_2024_2025.parquet"
    manifest = OUT / "manifest_pvb_history.json"

    pvb.to_csv(csv_plain, index=False, encoding="utf-8")
    with csv_plain.open("rb") as src, gzip.open(csv_gz, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    csv_plain.unlink()
    pvb.to_parquet(parquet, index=False, compression="zstd")

    payload = {
        "version": "0.1.1",
        "kind": "KBO historical PvB summary",
        "seasons": list(YEARS),
        "rows": int(len(pvb)),
        "columns": int(len(pvb.columns)),
        "key": ["season", "batter", "pitcher"],
        "source": "Naver Sports PBP via kbo_pbp_naver_sports",
        "generated_at_kst": datetime.now(KST).isoformat(),
        "files": {
            "csv_gz": csv_gz.name,
            "parquet": parquet.name,
            "sha256_csv_gz": sha256(csv_gz),
            "sha256_parquet": sha256(parquet),
        },
        "diagnostics": diagnostics,
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import kbo_pbp
from kbo_pbp import schedule

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"
STATUS_PATH = STATE_DIR / "status.json"
RUN_RESULT_PATH = STATE_DIR / "run_result.json"

ROUNDS = {schedule.REGULAR, *schedule.POSTSEASON}

NAVER_SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"

def _walk_game_dicts(obj: Any) -> list[dict[str, Any]]:
    """Recursively find schedule-like dicts containing gameId."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if "gameId" in obj:
            out.append(obj)
        for v in obj.values():
            out.extend(_walk_game_dicts(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk_game_dicts(v))
    return out

def fetch_live_daily_schedule(target: date) -> pd.DataFrame:
    """Fetch the date-scoped Naver schedule/status endpoint."""
    d = target.isoformat()
    params = {
        "fields": "basic,categoryName,statusNum",
        "upperCategoryId": "kbaseball",
        "fromDate": d,
        "toDate": d,
    }
    r = requests.get(
        NAVER_SCHEDULE_URL,
        params=params,
        headers={"User-Agent": "Mozilla/5.0 KBO-PBP-Updater/0.2"},
        timeout=20,
    )
    r.raise_for_status()
    games = _walk_game_dicts(r.json())
    if not games:
        raise RuntimeError("Naver date schedule returned no game objects.")

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for g in games:
        gid = str(g.get("gameId", ""))
        if gid and gid not in seen:
            seen.add(gid)
            unique.append(g)

    df = pd.json_normalize(unique, sep=".")
    for c in ("categoryName", "category.name"):
        if c in df.columns:
            s = df[c].astype(str)
            kbo_mask = s.str.contains("KBO|프로야구", case=False, regex=True, na=False)
            if kbo_mask.any():
                df = df[kbo_mask].copy()
            break
    return df

REQUIRED_COLUMNS = {
    "game_pk", "game_date", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "batter", "pitcher", "batter_name", "pitcher_name",
    "balls", "strikes", "outs_when_up",
    "home_score", "away_score",
    "pitch_result", "pitch_type", "pitch_name",
    "release_speed_kmh", "plate_x", "plate_z",
    "events", "post_home_score", "post_away_score",
}

@dataclass
class Result:
    version: str
    state: str
    validation: str
    target_date: str
    checked_at_kst: str
    first_check_kst: str | None = None
    has_1400_game: bool | None = None
    expected_games: int = 0
    final_games: int = 0
    expected_game_ids: list[str] | None = None
    found_game_ids: list[str] | None = None
    rows: int | None = None
    columns: int | None = None
    duplicate_stable_keys: int | None = None
    missing_columns: list[str] | None = None
    csv_gz: str | None = None
    parquet: str | None = None
    sha256_csv_gz: str | None = None
    sha256_parquet: str | None = None
    last_success_date: str | None = None
    reason: str | None = None
    should_publish: bool = False
    retry_next_hour: bool = False

def now_kst() -> datetime:
    return datetime.now(KST)

def load_previous_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_result(result: Result) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    RUN_RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def target_date_for_clock(now: datetime) -> date:
    if now.hour < 6:
        return now.date() - timedelta(days=1)
    return now.date()

def game_id_col(df: pd.DataFrame) -> str:
    for c in ("gameId", "game_id", "gamePk", "game_pk"):
        if c in df.columns:
            return c
    raise RuntimeError(f"Schedule game-id column not found: {list(df.columns)}")

def daily_schedule(df: pd.DataFrame, target: date) -> pd.DataFrame:
    gid = game_id_col(df)
    prefix = target.strftime("%Y%m%d")
    mask = df[gid].astype(str).str.startswith(prefix)
    if not mask.any():
        for c in ("gameDate", "game_date", "date"):
            if c in df.columns:
                vals = pd.to_datetime(df[c], errors="coerce").dt.date
                mask = vals == target
                break
    daily = df[mask].copy()
    if "roundCode" in daily.columns:
        daily = daily[daily["roundCode"].isin(ROUNDS)].copy()
    if "statusInfo" in daily.columns:
        cancelled = daily["statusInfo"].astype(str).str.contains("경기취소|취소", na=False)
        daily = daily[~cancelled].copy()
    return daily

def iter_scalar_strings(row: pd.Series) -> Iterable[str]:
    preferred = (
        "gameDateTime", "startDateTime", "startTime", "gameTime",
        "time", "startDate", "gameStartTime",
    )
    used: set[str] = set()
    for c in preferred:
        if c in row.index:
            used.add(c)
            yield str(row[c])
    for c, v in row.items():
        if c not in used and not isinstance(v, (dict, list, tuple, set)):
            yield str(v)

def has_1400_game(daily: pd.DataFrame) -> bool | None:
    if daily.empty:
        return False
    saw_time_like = False
    for _, row in daily.iterrows():
        for s in iter_scalar_strings(row):
            if re.search(r"\b(?:0?9|1[0-9]|2[0-3]):[0-5]\d\b", s):
                saw_time_like = True
            if re.search(r"(?<!\d)14:00(?::00)?(?!\d)", s) or s.strip() == "1400":
                return True
    return False if saw_time_like else None

def first_check_time(has_1400: bool | None) -> time:
    return time(17, 0) if has_1400 is not False else time(22, 0)

def final_mask(daily: pd.DataFrame) -> pd.Series:
    if daily.empty:
        return pd.Series([], dtype=bool, index=daily.index)

    if "statusNum" in daily.columns:
        n = pd.to_numeric(daily["statusNum"], errors="coerce")
        if n.notna().any():
            return n.eq(3)

    for c in ("gameStatusCode", "statusCode", "status", "gameStatus"):
        if c in daily.columns:
            s = daily[c].astype(str).str.upper()
            known = s.isin({"3", "RESULT", "END", "ENDED", "FINISHED", "FINAL", "COMPLETE", "COMPLETED"})
            if known.any():
                return known

    if "played" in daily.columns:
        return daily["played"].fillna(False).astype(bool)

    if "statusInfo" in daily.columns:
        return daily["statusInfo"].map(schedule.final_inning).notna()

    return pd.Series(False, index=daily.index)

def get_expected_game_ids(daily: pd.DataFrame) -> list[str]:
    if daily.empty:
        return []
    gid = game_id_col(daily)
    return sorted(daily[gid].astype(str).dropna().unique().tolist())

def get_final_game_ids(daily: pd.DataFrame) -> list[str]:
    if daily.empty:
        return []
    gid = game_id_col(daily)
    mask = final_mask(daily)
    if len(mask) != len(daily):
        return []
    return sorted(daily.loc[mask, gid].astype(str).dropna().unique().tolist())

def normalize_game_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date

def build_candidate(year: int) -> pd.DataFrame:
    kbo_pbp.download(year)
    df = kbo_pbp.season(year)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("kbo_pbp.season() did not return a DataFrame")
    return df

def validate_candidate(
    df: pd.DataFrame,
    target: date,
    expected_ids: list[str],
    previous: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}

    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    diagnostics["missing_columns"] = missing_cols
    if missing_cols:
        return False, diagnostics

    diagnostics["rows"] = int(len(df))
    diagnostics["columns"] = int(len(df.columns))

    stable = ["game_pk", "at_bat_number", "pitch_number"]
    dupes = int(df.duplicated(stable, keep=False).sum())
    diagnostics["duplicate_stable_keys"] = dupes
    if dupes:
        return False, diagnostics

    dates = normalize_game_date(df["game_date"])
    target_rows = df[dates == target]
    found_ids = sorted(target_rows["game_pk"].astype(str).unique().tolist())
    diagnostics["found_game_ids"] = found_ids

    missing_games = sorted(set(expected_ids) - set(found_ids))
    diagnostics["missing_game_ids"] = missing_games
    if missing_games:
        return False, diagnostics

    per_game_rows = target_rows.groupby(target_rows["game_pk"].astype(str)).size().to_dict()
    diagnostics["per_game_rows"] = {str(k): int(v) for k, v in per_game_rows.items()}
    if any(per_game_rows.get(g, 0) <= 0 for g in expected_ids):
        return False, diagnostics

    previous_rows = previous.get("rows")
    previous_success_date = previous.get("last_success_date")
    allow_regression = os.getenv("ALLOW_ROW_REGRESSION") == "1"
    if (
        previous_rows is not None
        and previous_success_date
        and int(len(df)) < int(previous_rows)
        and not allow_regression
    ):
        diagnostics["row_regression"] = {"previous": int(previous_rows), "candidate": int(len(df))}
        return False, diagnostics

    return True, diagnostics

def export_candidate(df: pd.DataFrame, year: int, target: date) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_plain = OUT_DIR / f"kbo_pbp_{year}_latest.csv"
    csv_gz = OUT_DIR / f"kbo_pbp_{year}_latest.csv.gz"
    parquet = OUT_DIR / f"kbo_pbp_{year}_latest.parquet"
    manifest = OUT_DIR / "manifest.json"

    df.to_csv(csv_plain, index=False, encoding="utf-8")
    with csv_plain.open("rb") as src, gzip.open(csv_gz, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    csv_plain.unlink()

    df.to_parquet(parquet, index=False, compression="zstd")

    payload = {
        "version": "0.2.0",
        "year": year,
        "through_date": target.isoformat(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "csv_gz": csv_gz.name,
        "parquet": parquet.name,
        "sha256_csv_gz": sha256(csv_gz),
        "sha256_parquet": sha256(parquet),
        "generated_at_kst": now_kst().isoformat(),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_gz, parquet, manifest

def run(target: date, force: bool = False) -> Result:
    now = now_kst()
    prev = load_previous_status()
    checked = now.isoformat()

    try:
        live_daily = fetch_live_daily_schedule(target)
        daily = daily_schedule(live_daily, target)
        schedule_source = "NAVER_DATE_STATUS"
    except Exception:
        season_schedule = schedule.fetch(target.year)
        daily = daily_schedule(season_schedule, target)
        schedule_source = "UPSTREAM_SEASON_SCHEDULE"

    expected_ids = get_expected_game_ids(daily)
    early = has_1400_game(daily)
    first = first_check_time(early)

    base = dict(
        version="0.2.0",
        target_date=target.isoformat(),
        checked_at_kst=checked,
        first_check_kst=first.strftime("%H:%M"),
        has_1400_game=early,
        expected_games=len(expected_ids),
        expected_game_ids=expected_ids,
        last_success_date=prev.get("last_success_date"),
    )

    if not expected_ids:
        return Result(
            state="NO_GAMES", validation="PASS", final_games=0,
            reason=f"[{schedule_source}] No eligible KBO regular/postseason games on target date.",
            retry_next_hour=False, **base
        )

    target_clock = datetime.combine(target, first, tzinfo=KST)
    if not force and now < target_clock:
        return Result(
            state="TOO_EARLY", validation="WAIT",
            final_games=int(final_mask(daily).sum()),
            reason=f"[{schedule_source}] First eligible check is {first.strftime('%H:%M')} KST.",
            retry_next_hour=True, **base
        )

    finals = final_mask(daily)
    final_count = int(finals.sum())

    # Naver's date-scoped endpoint can expose stale/non-final statusNum values
    # for already-completed historical dates. If that happens, cross-check the
    # upstream season schedule by game ID instead of blocking forever.
    if final_count < len(expected_ids) and schedule_source == "NAVER_DATE_STATUS":
        try:
            season_daily = daily_schedule(schedule.fetch(target.year), target)
            season_final_ids = set(get_final_game_ids(season_daily))
            expected_set = set(expected_ids)
            confirmed = expected_set & season_final_ids
            if expected_set and expected_set.issubset(season_final_ids):
                final_count = len(expected_ids)
                schedule_source = "NAVER_DATE_STATUS+UPSTREAM_FINAL_FALLBACK"
            elif len(confirmed) > final_count:
                final_count = len(confirmed)
        except Exception:
            pass

    if final_count < len(expected_ids):
        # Explicit historical backfills (--force) may receive stale/non-final
        # schedule status even though the games are already complete. In that
        # case, let the authoritative PBP validation prove completeness by
        # requiring every expected game ID and nonzero rows.
        historical_force = force and target < now.date()
        if not historical_force:
            return Result(
                state="GAMES_NOT_FINAL", validation="WAIT", final_games=final_count,
                reason=f"[{schedule_source}] Only {final_count}/{len(expected_ids)} expected games are final.",
                retry_next_hour=True, **base
            )
        final_count = len(expected_ids)
        schedule_source = schedule_source + "+FORCED_HISTORICAL_BACKFILL"

    if prev.get("last_success_date") == target.isoformat() and prev.get("validation") == "PASS" and not force:
        return Result(
            state="ALREADY_CURRENT", validation="PASS", final_games=final_count,
            rows=prev.get("rows"), columns=prev.get("columns"),
            found_game_ids=prev.get("found_game_ids"),
            duplicate_stable_keys=prev.get("duplicate_stable_keys"),
            missing_columns=prev.get("missing_columns"),
            csv_gz=prev.get("csv_gz"), parquet=prev.get("parquet"),
            sha256_csv_gz=prev.get("sha256_csv_gz"),
            sha256_parquet=prev.get("sha256_parquet"),
            last_success_date=target.isoformat(),
            reason=f"[{schedule_source}] Target date already validated successfully.",
            should_publish=False, retry_next_hour=False,
            **{k:v for k,v in base.items() if k != "last_success_date"}
        )

    try:
        df = build_candidate(target.year)
    except Exception as e:
        return Result(
            state="BUILD_ERROR", validation="FAIL", final_games=final_count,
            reason=f"{type(e).__name__}: {e}", retry_next_hour=True, **base
        )

    ok, diag = validate_candidate(df, target, expected_ids, prev)
    if not ok:
        return Result(
            state="VALIDATION_FAILED", validation="FAIL", final_games=final_count,
            found_game_ids=diag.get("found_game_ids"),
            rows=diag.get("rows"), columns=diag.get("columns"),
            duplicate_stable_keys=diag.get("duplicate_stable_keys"),
            missing_columns=diag.get("missing_columns"),
            reason=json.dumps(diag, ensure_ascii=False),
            retry_next_hour=True, **base
        )

    csv_gz, parquet, _manifest = export_candidate(df, target.year, target)
    return Result(
        state="UPDATED", validation="PASS", final_games=final_count,
        found_game_ids=diag["found_game_ids"],
        rows=diag["rows"], columns=diag["columns"],
        duplicate_stable_keys=diag["duplicate_stable_keys"],
        missing_columns=diag["missing_columns"],
        csv_gz=csv_gz.name, parquet=parquet.name,
        sha256_csv_gz=sha256(csv_gz),
        sha256_parquet=sha256(parquet),
        last_success_date=target.isoformat(),
        reason=f"[{schedule_source}] All expected games present; dataset validated and exported.",
        should_publish=True, retry_next_hour=False,
        **{k:v for k,v in base.items() if k != "last_success_date"}
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="Target KST date YYYY-MM-DD. Default: today, or previous day before 06:00 KST.")
    p.add_argument("--force", action="store_true", help="Ignore first-check gate and rebuild even if date already succeeded.")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else target_date_for_clock(now_kst())
    result = run(target, force=args.force)
    save_result(result)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

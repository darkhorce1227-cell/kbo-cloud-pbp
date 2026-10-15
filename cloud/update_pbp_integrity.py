from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

import pandas as pd

import update_pbp as core
import update_pbp_forward as forward
import update_pbp_guarded as guarded
import update_pbp_resilient as resilient

_ORIGINAL_GUARDED_VALIDATE = guarded._guarded_validate
_ORIGINAL_GUARDED_EXPORT = guarded._guarded_export


def _encoded_game_date(game_id: str, year: int) -> date | None:
    """Recover the actual game date from a KBO/Naver game ID when possible.

    Regular-season IDs start YYYYMMDD. Current postseason IDs can use a
    non-date prefix but still carry MMDD at positions 4:8 and the season year
    in the final four characters (e.g. 44441014OBLG02026 -> 2026-10-14).
    """
    s = str(game_id)
    if len(s) >= 8 and s[:8].isdigit():
        try:
            d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
            if d.year == year:
                return d
        except ValueError:
            pass
    if len(s) >= 8 and len(s) >= 4 and s[4:8].isdigit() and s[-4:].isdigit():
        try:
            y = int(s[-4:])
            if y == year:
                return date(y, int(s[4:6]), int(s[6:8]))
        except ValueError:
            pass
    return None


def _date_mismatches(df: pd.DataFrame, year: int) -> list[dict[str, str]]:
    if "game_pk" not in df.columns or "game_date" not in df.columns:
        return [{"game_pk": "<schema>", "expected": "game_pk+game_date", "actual": "missing"}]
    unique = df[["game_pk", "game_date"]].drop_duplicates()
    out: list[dict[str, str]] = []
    for row in unique.itertuples(index=False):
        expected = _encoded_game_date(str(row.game_pk), year)
        if expected is None:
            continue
        actual_ts = pd.to_datetime(row.game_date, errors="coerce")
        actual = None if pd.isna(actual_ts) else actual_ts.date()
        if actual != expected:
            out.append({
                "game_pk": str(row.game_pk),
                "expected": expected.isoformat(),
                "actual": "NaT" if actual is None else actual.isoformat(),
            })
    return out


def _validate_with_date_integrity(
    df: pd.DataFrame,
    target: date,
    expected_ids: list[str],
    previous: dict[str, object],
):
    ok, diag = _ORIGINAL_GUARDED_VALIDATE(df, target, expected_ids, previous)
    mismatches = _date_mismatches(df, target.year)
    diag["game_date_mismatch_count"] = len(mismatches)
    diag["game_date_mismatches"] = mismatches[:50]
    if mismatches:
        return False, diag
    return ok, diag


def _export_with_date_integrity(df: pd.DataFrame, year: int, target: date):
    mismatches = _date_mismatches(df, year)
    if mismatches:
        raise RuntimeError(
            "game_date integrity failure before export: "
            + json.dumps(mismatches[:50], ensure_ascii=False)
        )
    return _ORIGINAL_GUARDED_EXPORT(df, year, target)


def _date_safe_direct_frames(
    fallback_target: date,
    year: int,
    expected_ids: list[str],
):
    """Direct-fetch games without stamping an unrelated repair target date.

    Continuity recovery can fetch many prior game IDs while rebuilding a newer
    target. Each recovered game therefore receives the date encoded in its own
    game ID; the requested target is only a fallback for an unknown ID format.
    """
    try:
        cached = resilient.kbo_pbp.load_schedule(year)
    except Exception:
        cached = None

    live_cache: dict[date, pd.DataFrame | None] = {}
    frames: list[pd.DataFrame] = []
    report: dict[str, object] = {}

    for gid in expected_ids:
        game_target = _encoded_game_date(str(gid), year) or fallback_target
        row = forward._row_for_id(cached, str(gid))
        metadata_source = "cached_season_schedule"

        if row is None:
            if game_target not in live_cache:
                try:
                    live_cache[game_target] = core.fetch_live_daily_schedule(game_target)
                except Exception:
                    live_cache[game_target] = None
            row = forward._row_for_id(live_cache[game_target], str(gid))
            metadata_source = "live_daily_schedule"

        meta: dict[str, object] = row.to_dict() if row is not None else {}
        meta["gameId"] = str(gid)
        meta["gameDate"] = game_target.isoformat()

        home = forward._meta_value(meta, "homeTeamCode")
        away = forward._meta_value(meta, "awayTeamCode")
        if home is None or away is None:
            away_gid, home_gid = forward._team_codes_from_game_id(str(gid))
            away = away if away is not None else away_gid
            home = home if home is not None else home_gid
            metadata_source += "+game_id_team_fallback"

        if home is None or away is None:
            raise RuntimeError(
                f"game metadata incomplete after date-safe fallback: {gid}; "
                f"date={game_target} home={home!r} away={away!r}"
            )

        meta["homeTeamCode"] = str(home)
        meta["awayTeamCode"] = str(away)

        game, final_inning = resilient._direct_fetch_game(year, str(gid))
        frame = resilient.statcast.game_frame(
            game,
            resilient.statcast.meta_from_schedule(meta),
        )
        if frame.empty:
            raise RuntimeError(f"direct relay produced empty frame: {gid}")

        frames.append(frame)
        report[str(gid)] = {
            "game_end": True,
            "final_inning": final_inning,
            "rows": int(len(frame)),
            "metadata_source": metadata_source,
            "game_date": game_target.isoformat(),
            "home_team": str(home),
            "away_team": str(away),
        }

    return frames, report


def _install_integrity_patches(target: date) -> None:
    # Patch before forward.run() because its continuity-repair branch executes
    # before the old target-scoped metadata patch was installed.
    resilient._direct_daily_frames = (
        lambda year, expected_ids: _date_safe_direct_frames(
            target, year, list(expected_ids)
        )
    )
    forward._direct_daily_frames_with_live_meta = _date_safe_direct_frames
    guarded._guarded_validate = _validate_with_date_integrity
    guarded._guarded_export = _export_with_date_integrity


def run(target: date, force: bool = False) -> core.Result:
    _install_integrity_patches(target)
    result = forward.run(target, force=force)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="Target KST date YYYY-MM-DD")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.date:
        target = date.fromisoformat(args.date)
    else:
        automatic = core.target_date_for_clock(core.now_kst())
        target = forward._pending_prior_target(automatic) or automatic
    result = run(target, force=args.force)
    core.save_result(result)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

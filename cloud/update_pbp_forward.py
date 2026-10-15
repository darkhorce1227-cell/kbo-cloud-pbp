from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import update_pbp as core
import update_pbp_resilient as resilient
import update_pbp_guarded as guarded


RECOVERY_LOOKBACK_DAYS = 7


def _patch_continuity_guard() -> None:
    """Apply cumulative continuity recovery/validation without hijacking target-date flow."""
    core.build_candidate = guarded._recover_candidate
    core.validate_candidate = guarded._guarded_validate
    core.export_candidate = guarded._guarded_export


def _row_for_id(df: pd.DataFrame | None, gid: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    try:
        gid_col = core.game_id_col(df)
    except Exception:
        return None
    hit = df[df[gid_col].astype(str) == str(gid)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _meta_value(meta: dict[str, object], key: str) -> object | None:
    """Read an exact or json_normalize-suffixed metadata field."""
    candidates = [key, f"basic.{key}"]
    for k in candidates:
        value = meta.get(k)
        if value is not None and not pd.isna(value):
            return value
    suffix = f".{key}"
    for k, value in meta.items():
        if str(k).endswith(suffix) and value is not None and not pd.isna(value):
            return value
    return None


def _team_codes_from_game_id(gid: str) -> tuple[str | None, str | None]:
    """Fallback for Naver KBO IDs: chars 8:10 away, 10:12 home.

    This also works for postseason IDs whose first eight characters are not a
    literal YYYYMMDD (for example 44441014OBLG02026).
    """
    s = str(gid)
    if len(s) < 12:
        return None, None
    away, home = s[8:10], s[10:12]
    if away.isalpha() and home.isalpha():
        return away, home
    return None, None


def _direct_daily_frames_with_live_meta(
    target: date,
    year: int,
    expected_ids: list[str],
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    """Direct-fetch games while allowing live date-scoped schedule metadata.

    Postseason game IDs can appear in Naver's live date endpoint before they are
    present in the upstream package's cached season schedule. The old fallback
    failed closed at that metadata lookup even though the relay itself was
    available. Prefer cached metadata when present, otherwise use the same live
    schedule endpoint that produced expected_ids. Force target gameDate and, if
    necessary, recover two-letter team codes from the game ID.
    """
    try:
        cached = resilient.kbo_pbp.load_schedule(year)
    except Exception:
        cached = None

    try:
        live = core.fetch_live_daily_schedule(target)
    except Exception:
        live = None

    frames: list[pd.DataFrame] = []
    report: dict[str, object] = {}

    for gid in expected_ids:
        row = _row_for_id(cached, gid)
        metadata_source = "cached_season_schedule"
        if row is None:
            row = _row_for_id(live, gid)
            metadata_source = "live_daily_schedule"

        meta: dict[str, object] = row.to_dict() if row is not None else {}
        meta["gameId"] = str(gid)
        meta["gameDate"] = target.isoformat()

        home = _meta_value(meta, "homeTeamCode")
        away = _meta_value(meta, "awayTeamCode")
        if home is None or away is None:
            away_gid, home_gid = _team_codes_from_game_id(str(gid))
            away = away if away is not None else away_gid
            home = home if home is not None else home_gid
            metadata_source += "+game_id_team_fallback"

        if home is None or away is None:
            raise RuntimeError(
                f"game metadata incomplete after live fallback: {gid}; "
                f"home={home!r} away={away!r}"
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
            "home_team": str(home),
            "away_team": str(away),
        }

    return frames, report


def _patch_postseason_live_metadata(target: date) -> None:
    """Patch resilient direct fetch with target-aware live metadata recovery."""
    resilient._direct_daily_frames = (
        lambda year, expected_ids: _direct_daily_frames_with_live_meta(
            target, year, list(expected_ids)
        )
    )


def _pending_prior_target(auto_target: date) -> date | None:
    """Return the newest recent dated audit that still needs a real rebuild."""
    state_dir = Path(core.STATE_DIR)
    for days in range(1, RECOVERY_LOOKBACK_DAYS + 1):
        candidate = auto_target - timedelta(days=days)
        path = state_dir / f"postgame_score_audit_{candidate.isoformat()}.json"
        if not path.exists():
            continue
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(audit.get("expected_games") or 0) <= 0:
            continue
        if audit.get("validation") != "PASS":
            return candidate
    return None


def _direct_recover_validation_failure(target: date, result: core.Result) -> core.Result:
    """Recover target-date games when schedule says FINAL but upstream season() omitted them.

    The older resilient layer only attempted direct relay when the schedule itself
    was non-final. This closes the second hole: FINAL schedule + stale upstream
    playable set + missing target games.
    """
    expected_ids = list(result.expected_game_ids or [])
    if not expected_ids:
        return result

    try:
        df = guarded._recover_candidate(target.year)
        df, direct_report = resilient._augment_candidate_with_direct_games(df, target, expected_ids)
    except Exception as e:
        result.reason = (
            f"{result.reason}; forward direct recovery failed: {type(e).__name__}: {e}"
        )
        result.retry_next_hour = True
        return result

    prev = core.load_previous_status()
    ok, diag = guarded._guarded_validate(df, target, expected_ids, prev)
    if not ok:
        result.reason = (
            "[FORWARD_DIRECT_RECOVERY_FAILED] "
            + json.dumps(diag, ensure_ascii=False)
            + "; direct="
            + json.dumps(direct_report, ensure_ascii=False)
        )
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        result.retry_next_hour = True
        return result

    terminal_ok, terminal_report = resilient._terminal_audit(
        df, target, expected_ids, direct_report
    )
    if not terminal_ok:
        result.state = "GAMES_NOT_FINAL"
        result.validation = "WAIT"
        result.reason = (
            "[FORWARD_DIRECT_RECOVERY_WAIT] terminal proof failed: "
            + json.dumps(terminal_report, ensure_ascii=False)
        )
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        result.should_publish = False
        result.retry_next_hour = True
        return result

    csv_gz, parquet, _manifest = guarded._guarded_export(df, target.year, target)
    return core.Result(
        version="0.5.0",
        state="UPDATED",
        validation="PASS",
        target_date=target.isoformat(),
        checked_at_kst=core.now_kst().isoformat(),
        first_check_kst=result.first_check_kst,
        has_1400_game=result.has_1400_game,
        expected_games=len(expected_ids),
        final_games=len(expected_ids),
        expected_game_ids=expected_ids,
        found_game_ids=diag.get("found_game_ids"),
        rows=diag.get("rows"),
        columns=diag.get("columns"),
        duplicate_stable_keys=diag.get("duplicate_stable_keys"),
        missing_columns=diag.get("missing_columns"),
        csv_gz=csv_gz.name,
        parquet=parquet.name,
        sha256_csv_gz=core.sha256(csv_gz),
        sha256_parquet=core.sha256(parquet),
        last_success_date=target.isoformat(),
        reason=(
            "[FORWARD_DIRECT_GAME_END_RECOVERY] Target-date schedule was final or usable, "
            "but the upstream rebuilt season omitted one or more expected games. Missing target "
            "games were fetched directly by gameId using cached or live date-scoped metadata and "
            "accepted only with explicit Naver GAME_END(type=99); cumulative game_pk continuity "
            "and normal validation passed."
        ),
        should_publish=True,
        retry_next_hour=False,
    )


def run(target: date, force: bool = False) -> core.Result:
    index = guarded._load_index(target.year)
    through_s = str(index.get("through_date") or "")

    # Maintenance/backfill of the current or an older checkpoint may repair the
    # previous release in place. A NEWER target must NEVER be short-circuited by
    # old-date continuity repair; continuity is recovered inside its candidate.
    if through_s:
        through = date.fromisoformat(through_s)
        if target <= through:
            repair = guarded._repair_previous_release_if_needed(target.year)
            if repair is not None:
                return repair

    _patch_continuity_guard()
    _patch_postseason_live_metadata(target)
    result = resilient.run(target, force=force)

    # Close the FINAL-schedule / stale-upstream hole. The resilient module
    # already handles GAMES_NOT_FINAL via direct GAME_END fetch; this branch
    # handles VALIDATION_FAILED caused by missing target-date game IDs.
    if result.state == "VALIDATION_FAILED" and result.validation == "FAIL":
        return _direct_recover_validation_failure(target, result)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="Target KST date YYYY-MM-DD. Default: today, or a recent unresolved dated score-audit before today.")
    p.add_argument("--force", action="store_true", help="Ignore first-check gate and rebuild even if date already succeeded.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.date:
        target = date.fromisoformat(args.date)
    else:
        automatic = core.target_date_for_clock(core.now_kst())
        target = _pending_prior_target(automatic) or automatic
    result = run(target, force=args.force)
    core.save_result(result)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

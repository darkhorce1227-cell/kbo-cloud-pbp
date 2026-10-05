from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

import update_pbp as core
import update_pbp_resilient as resilient
import update_pbp_guarded as guarded


def _patch_continuity_guard() -> None:
    """Apply cumulative continuity recovery/validation without hijacking target-date flow."""
    core.build_candidate = guarded._recover_candidate
    core.validate_candidate = guarded._guarded_validate
    core.export_candidate = guarded._guarded_export


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
        version="0.4.0",
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
            "games were fetched directly by gameId and accepted only with explicit Naver "
            "GAME_END(type=99); cumulative game_pk continuity and normal validation passed."
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
    result = resilient.run(target, force=force)

    # Close the FINAL-schedule / stale-upstream hole. The resilient module
    # already handles GAMES_NOT_FINAL via direct GAME_END fetch; this branch
    # handles VALIDATION_FAILED caused by missing target-date game IDs.
    if result.state == "VALIDATION_FAILED" and result.validation == "FAIL":
        return _direct_recover_validation_failure(target, result)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="Target KST date YYYY-MM-DD. Default: today, or previous day before 06:00 KST.")
    p.add_argument("--force", action="store_true", help="Ignore first-check gate and rebuild even if date already succeeded.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else core.target_date_for_clock(core.now_kst())
    result = run(target, force=args.force)
    core.save_result(result)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

import pandas as pd

import update_pbp as core


OUT_EVENTS = {
    "field_out",
    "strikeout",
    "strikeout_double_play",
    "double_play",
    "triple_play",
    "force_out",
    "grounded_into_double_play",
    "sac_fly",
    "sac_bunt",
    "caught_stealing",
    "pickoff_caught_stealing",
}


def _is_terminal_game(game: pd.DataFrame) -> tuple[bool, dict[str, object]]:
    """Conservatively prove a game is final from the PBP itself.

    This fallback is used only when Naver's date-scoped schedule status is stale.
    It deliberately fails closed: if the final state cannot be proven from the
    last recorded PA, the updater stays in WAIT rather than publishing a partial
    game.
    """
    if game.empty:
        return False, {"reason": "no_rows"}

    work = game.copy()
    work["_inning"] = pd.to_numeric(work["inning"], errors="coerce")
    work["_ab"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
    work["_pitch"] = pd.to_numeric(work["pitch_number"], errors="coerce")
    work = work.sort_values(["_inning", "_ab", "_pitch"], kind="stable")
    last = work.iloc[-1]

    try:
        inning = int(last["_inning"])
    except Exception:
        return False, {"reason": "bad_inning"}

    half = str(last.get("inning_topbot", "")).strip().lower()
    try:
        home = int(float(last.get("post_home_score")))
        away = int(float(last.get("post_away_score")))
    except Exception:
        return False, {"reason": "bad_score"}

    outs_before = pd.to_numeric(pd.Series([last.get("outs_when_up")]), errors="coerce").iloc[0]
    event = str(last.get("events", "")).strip().lower()
    third_out = bool(pd.notna(outs_before) and int(outs_before) >= 2 and event in OUT_EVENTS)

    diag = {
        "inning": inning,
        "half": half,
        "home": home,
        "away": away,
        "outs_before": None if pd.isna(outs_before) else int(outs_before),
        "event": event,
        "third_out": third_out,
    }

    if inning < 9:
        diag["reason"] = "before_regulation_end"
        return False, diag
    if home == away:
        diag["reason"] = "tied_after_last_play"
        return False, diag

    is_top = half.startswith("top") or half in {"초", "t"}
    is_bottom = half.startswith("bot") or half.startswith("bottom") or half in {"말", "b"}

    if is_top:
        # A game may end after the top half only when the home team already leads
        # and the third out of that half has been recorded.
        ok = home > away and third_out
        diag["reason"] = "home_lead_after_top_with_3rd_out" if ok else "top_half_not_terminal"
        return ok, diag

    if is_bottom:
        if home > away:
            # Home lead in B9+ is terminal either by walk-off or by the third out
            # when the home team entered the half already ahead.
            diag["reason"] = "home_lead_bottom_9plus"
            return True, diag
        ok = away > home and third_out
        diag["reason"] = "away_lead_after_bottom_3rd_out" if ok else "bottom_half_not_terminal"
        return ok, diag

    diag["reason"] = "unknown_half"
    return False, diag


def _terminal_audit(df: pd.DataFrame, target: date, expected_ids: list[str]) -> tuple[bool, dict[str, object]]:
    dates = core.normalize_game_date(df["game_date"])
    target_rows = df[dates == target]
    report: dict[str, object] = {}
    all_final = True
    for gid in expected_ids:
        g = target_rows[target_rows["game_pk"].astype(str) == str(gid)]
        ok, diag = _is_terminal_game(g)
        report[str(gid)] = {"final": ok, **diag}
        all_final = all_final and ok
    return all_final, report


def run(target: date, force: bool = False) -> core.Result:
    result = core.run(target, force=force)
    if result.state != "GAMES_NOT_FINAL" or result.validation != "WAIT":
        return result

    # The schedule endpoint can remain stale even after a completed slate. Build
    # the candidate and require both the normal dataset validation and an
    # independent terminal-play proof for every expected game before publishing.
    expected_ids = list(result.expected_game_ids or [])
    if not expected_ids:
        return result

    try:
        df = core.build_candidate(target.year)
    except Exception as e:
        result.reason = f"{result.reason}; PBP terminal fallback build failed: {type(e).__name__}: {e}"
        return result

    prev = core.load_previous_status()
    ok, diag = core.validate_candidate(df, target, expected_ids, prev)
    if not ok:
        result.reason = f"{result.reason}; PBP terminal fallback candidate incomplete: {json.dumps(diag, ensure_ascii=False)}"
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        return result

    terminal_ok, terminal_report = _terminal_audit(df, target, expected_ids)
    if not terminal_ok:
        result.reason = f"{result.reason}; PBP terminal fallback not proven: {json.dumps(terminal_report, ensure_ascii=False)}"
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        return result

    csv_gz, parquet, _manifest = core.export_candidate(df, target.year, target)
    return core.Result(
        version="0.2.1",
        state="UPDATED",
        validation="PASS",
        target_date=target.isoformat(),
        checked_at_kst=core.now_kst().isoformat(),
        first_check_kst=result.first_check_kst,
        has_1400_game=result.has_1400_game,
        expected_games=len(expected_ids),
        final_games=len(expected_ids),
        expected_game_ids=expected_ids,
        found_game_ids=diag["found_game_ids"],
        rows=diag["rows"],
        columns=diag["columns"],
        duplicate_stable_keys=diag["duplicate_stable_keys"],
        missing_columns=diag["missing_columns"],
        csv_gz=csv_gz.name,
        parquet=parquet.name,
        sha256_csv_gz=core.sha256(csv_gz),
        sha256_parquet=core.sha256(parquet),
        last_success_date=target.isoformat(),
        reason="[PBP_TERMINAL_FALLBACK] Schedule status was stale; every expected game passed terminal-play proof and dataset validation.",
        should_publish=True,
        retry_next_hour=False,
    )


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

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

import pandas as pd
import kbo_pbp
from kbo_pbp import naver, relay, statcast, storage

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

MAX_INNINGS = 15


def _relay_has_game_end(payload: dict) -> bool:
    """Return True only when Naver relay itself contains GAME_END(type=99)."""
    for block in payload.get("textRelays") or []:
        for event in block.get("textOptions") or []:
            try:
                if int(event.get("type", -1)) == int(relay.GAME_END):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _direct_fetch_game(year: int, game_id: str) -> tuple[dict[int, dict], int]:
    """Fetch a game inning-by-inning without relying on schedule.playable().

    Fail closed unless an explicit Naver GAME_END(type=99) event is observed.
    The completed relay is cached in the same format used by upstream kbo_pbp.
    """
    relays: dict[int, dict] = {}
    final_inning: int | None = None
    for inning in range(1, MAX_INNINGS + 1):
        payload = naver.fetch_relay(game_id, inning)
        relays[inning] = payload
        if _relay_has_game_end(payload):
            final_inning = inning
            break

    if final_inning is None:
        raise RuntimeError(f"GAME_END not observed through inning {MAX_INNINGS}: {game_id}")

    storage.write_json(storage.game_file(year, game_id), relays)
    return relays, final_inning


def _direct_daily_frames(year: int, expected_ids: list[str]) -> tuple[list[pd.DataFrame], dict[str, object]]:
    """Direct-fetch expected games and build Statcast-style frames.

    Metadata comes from the cached season schedule row, which already contains
    the expected game IDs even when its final-status field is stale.
    """
    schedule_df = kbo_pbp.load_schedule(year)
    gid_col = core.game_id_col(schedule_df)
    by_id = schedule_df.set_index(schedule_df[gid_col].astype(str), drop=False)

    frames: list[pd.DataFrame] = []
    report: dict[str, object] = {}
    for gid in expected_ids:
        if gid not in by_id.index:
            raise RuntimeError(f"game metadata missing from season schedule: {gid}")
        row = by_id.loc[gid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        game, final_inning = _direct_fetch_game(year, gid)
        meta_source = row.to_dict()
        meta_source["gameId"] = gid
        frame = statcast.game_frame(game, statcast.meta_from_schedule(meta_source))
        if frame.empty:
            raise RuntimeError(f"direct relay produced empty frame: {gid}")
        frames.append(frame)
        report[gid] = {
            "game_end": True,
            "final_inning": final_inning,
            "rows": int(len(frame)),
        }
    return frames, report


def _augment_candidate_with_direct_games(
    df: pd.DataFrame,
    target: date,
    expected_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    dates = core.normalize_game_date(df["game_date"])
    present = set(df.loc[dates == target, "game_pk"].astype(str).unique().tolist())
    missing = [gid for gid in expected_ids if gid not in present]
    if not missing:
        return df, {"direct_fetch": [], "already_present": sorted(present & set(expected_ids))}

    frames, report = _direct_daily_frames(target.year, missing)
    # Defensive removal protects against a partially present game before concat.
    keep = ~df["game_pk"].astype(str).isin(missing)
    out = pd.concat([df.loc[keep], *frames], ignore_index=True, sort=False)
    return out, {
        "direct_fetch": missing,
        "already_present": sorted(present & set(expected_ids)),
        "games": report,
    }


def _is_terminal_game(game: pd.DataFrame) -> tuple[bool, dict[str, object]]:
    """Conservatively prove a final state from the parsed pitch rows."""
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
        ok = home > away and third_out
        diag["reason"] = "home_lead_after_top_with_3rd_out" if ok else "top_half_not_terminal"
        return ok, diag

    if is_bottom:
        if home > away:
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

    expected_ids = list(result.expected_game_ids or [])
    if not expected_ids:
        return result

    # First get the normal cached season. If the upstream package's
    # schedule.playable() is stale too, explicitly fetch the missing game IDs.
    try:
        df = core.build_candidate(target.year)
        df, direct_report = _augment_candidate_with_direct_games(df, target, expected_ids)
    except Exception as e:
        result.reason = f"{result.reason}; direct PBP fallback failed: {type(e).__name__}: {e}"
        return result

    prev = core.load_previous_status()
    ok, diag = core.validate_candidate(df, target, expected_ids, prev)
    if not ok:
        result.reason = (
            f"{result.reason}; direct PBP fallback candidate incomplete: "
            f"{json.dumps(diag, ensure_ascii=False)}; direct={json.dumps(direct_report, ensure_ascii=False)}"
        )
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        return result

    terminal_ok, terminal_report = _terminal_audit(df, target, expected_ids)
    if not terminal_ok:
        result.reason = (
            f"{result.reason}; direct GAME_END was observed but parsed terminal audit failed: "
            f"{json.dumps(terminal_report, ensure_ascii=False)}"
        )
        result.found_game_ids = diag.get("found_game_ids")
        result.rows = diag.get("rows")
        result.columns = diag.get("columns")
        result.duplicate_stable_keys = diag.get("duplicate_stable_keys")
        result.missing_columns = diag.get("missing_columns")
        return result

    csv_gz, parquet, _manifest = core.export_candidate(df, target.year, target)
    return core.Result(
        version="0.2.2",
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
        reason=(
            "[DIRECT_RELAY_GAME_END_FALLBACK] Schedule status was stale; missing expected games "
            "were fetched directly by game ID, each exposed Naver GAME_END(type=99), and every "
            "game passed parsed terminal-state plus dataset validation."
        ),
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

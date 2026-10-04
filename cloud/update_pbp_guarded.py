from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

import update_pbp as core
import update_pbp_resilient as base


def _index_path(year: int) -> Path:
    return core.STATE_DIR / f"validated_game_pks_{year}.json"


def _game_ids(df: pd.DataFrame) -> set[str]:
    if "game_pk" not in df.columns:
        return set()
    return set(df["game_pk"].dropna().astype(str).unique().tolist())


def _digest(ids: set[str] | list[str]) -> str:
    ordered = sorted(set(str(x) for x in ids))
    payload = ("\n".join(ordered) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_index(year: int) -> dict[str, Any]:
    path = _index_path(year)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if int(data.get("year", year)) != year:
        return {}
    pks = [str(x) for x in data.get("game_pks", []) if str(x)]
    data["game_pks"] = pks
    return data


def _write_index(df: pd.DataFrame, year: int, through: date) -> dict[str, Any]:
    ids = sorted(_game_ids(df))
    payload = {
        "version": "1.0",
        "year": year,
        "through_date": through.isoformat(),
        "bootstrap_minimum": False,
        "game_count": len(ids),
        "game_pk_sha256": _digest(ids),
        "game_pks": ids,
    }
    path = _index_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _replace_games(
    df: pd.DataFrame,
    year: int,
    game_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not game_ids:
        return df, {"direct_fetch": [], "games": {}}
    frames, report = base._direct_daily_frames(year, game_ids)
    keep = ~df["game_pk"].astype(str).isin(game_ids)
    out = pd.concat([df.loc[keep], *frames], ignore_index=True, sort=False)
    return out, {"direct_fetch": game_ids, "games": report}


_ORIGINAL_BUILD = core.build_candidate
_ORIGINAL_VALIDATE = core.validate_candidate
_ORIGINAL_EXPORT = core.export_candidate


def _recover_candidate(year: int) -> pd.DataFrame:
    df = _ORIGINAL_BUILD(year)
    index = _load_index(year)
    required = set(index.get("game_pks", []))
    present = _game_ids(df)
    missing = sorted(required - present)
    if missing:
        df, _ = _replace_games(df, year, missing)
    return df


def _guarded_validate(
    df: pd.DataFrame,
    target: date,
    expected_ids: list[str],
    previous: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    ok, diag = _ORIGINAL_VALIDATE(df, target, expected_ids, previous)
    candidate_ids = _game_ids(df)
    index = _load_index(target.year)
    required = set(index.get("game_pks", []))
    missing_prior = sorted(required - candidate_ids)
    diag["candidate_game_count"] = len(candidate_ids)
    diag["prior_validated_game_count"] = len(required)
    diag["missing_prior_game_pks"] = missing_prior
    diag["candidate_game_pk_sha256"] = _digest(candidate_ids)
    if missing_prior:
        return False, diag
    return ok, diag


def _guarded_export(
    df: pd.DataFrame,
    year: int,
    target: date,
):
    csv_gz, parquet, manifest = _ORIGINAL_EXPORT(df, year, target)
    index = _write_index(df, year, target)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["game_count"] = index["game_count"]
    payload["game_pk_sha256"] = index["game_pk_sha256"]
    payload["continuity_guard"] = "previous_validated_game_pk_subset"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_gz, parquet, manifest


def _repair_previous_release_if_needed(year: int) -> core.Result | None:
    index = _load_index(year)
    required = set(index.get("game_pks", []))
    if not required:
        return None

    raw = _ORIGINAL_BUILD(year)
    present = _game_ids(raw)
    missing = sorted(required - present)
    if not missing:
        return None

    through_s = str(index.get("through_date") or "")
    if not through_s:
        return None
    through = date.fromisoformat(through_s)

    repaired, recovery_report = _replace_games(raw, year, missing)

    # Re-fetch the checkpoint day's games too. This preserves explicit
    # GAME_END(type=99) proof for legal tie/called endings during repair.
    prefix = through.strftime("%Y%m%d")
    checkpoint_ids = sorted(gid for gid in required if gid.startswith(prefix))
    checkpoint_report: dict[str, object] = {"direct_fetch": [], "games": {}}
    if checkpoint_ids:
        repaired, checkpoint_report = _replace_games(repaired, year, checkpoint_ids)

    expected_ids = checkpoint_ids
    ok, diag = _ORIGINAL_VALIDATE(repaired, through, expected_ids, {})
    candidate_ids = _game_ids(repaired)
    missing_after = sorted(required - candidate_ids)
    if missing_after:
        ok = False
        diag["missing_prior_game_pks"] = missing_after

    terminal_ok = True
    terminal_report: dict[str, object] = {}
    if expected_ids:
        terminal_ok, terminal_report = base._terminal_audit(
            repaired,
            through,
            expected_ids,
            checkpoint_report,
        )

    if not ok or not terminal_ok:
        return core.Result(
            version="0.3.0",
            state="CONTINUITY_REPAIR_FAILED",
            validation="FAIL",
            target_date=through.isoformat(),
            checked_at_kst=core.now_kst().isoformat(),
            expected_games=len(expected_ids),
            final_games=0,
            expected_game_ids=expected_ids,
            found_game_ids=diag.get("found_game_ids"),
            rows=diag.get("rows"),
            columns=diag.get("columns"),
            duplicate_stable_keys=diag.get("duplicate_stable_keys"),
            missing_columns=diag.get("missing_columns"),
            last_success_date=through.isoformat(),
            reason=(
                "Continuity repair failed. "
                f"missing_before={missing}; diagnostics={json.dumps(diag, ensure_ascii=False)}; "
                f"terminal={json.dumps(terminal_report, ensure_ascii=False)}; "
                f"recovery={json.dumps(recovery_report, ensure_ascii=False)}"
            ),
            should_publish=False,
            retry_next_hour=False,
        )

    csv_gz, parquet, _manifest = _guarded_export(repaired, year, through)
    full_ids = _game_ids(repaired)
    return core.Result(
        version="0.3.0",
        state="CONTINUITY_REPAIRED",
        validation="PASS",
        target_date=through.isoformat(),
        checked_at_kst=core.now_kst().isoformat(),
        expected_games=len(expected_ids),
        final_games=len(expected_ids),
        expected_game_ids=expected_ids,
        found_game_ids=diag.get("found_game_ids"),
        rows=int(len(repaired)),
        columns=int(len(repaired.columns)),
        duplicate_stable_keys=int(repaired.duplicated(
            ["game_pk", "at_bat_number", "pitch_number"], keep=False
        ).sum()),
        missing_columns=diag.get("missing_columns"),
        csv_gz=csv_gz.name,
        parquet=parquet.name,
        sha256_csv_gz=core.sha256(csv_gz),
        sha256_parquet=core.sha256(parquet),
        last_success_date=through.isoformat(),
        reason=(
            "[GAME_PK_CONTINUITY_REPAIR] Recovered previously validated games missing "
            f"from the rebuilt candidate: {missing}. Candidate now preserves "
            f"{len(required)} required prior game_pk values and contains {len(full_ids)} games total."
        ),
        should_publish=True,
        retry_next_hour=False,
    )


def run(target: date, force: bool = False) -> core.Result:
    repair = _repair_previous_release_if_needed(target.year)
    if repair is not None:
        return repair

    core.build_candidate = _recover_candidate
    core.validate_candidate = _guarded_validate
    core.export_candidate = _guarded_export

    result = base.run(target, force=force)
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

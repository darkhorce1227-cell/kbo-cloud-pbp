from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"
RUN_RESULT_PATH = STATE_DIR / "run_result.json"
STATUS_PATH = STATE_DIR / "status.json"
LATEST_AUDIT_PATH = STATE_DIR / "postgame_score_audit.json"

NAVER_SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
STABLE_KEYS = ["game_pk", "at_bat_number", "pitch_number"]


def now_kst() -> datetime:
    return datetime.now(KST)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_run_result() -> dict[str, Any]:
    for p in (RUN_RESULT_PATH, STATUS_PATH):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise RuntimeError("state/run_result.json and state/status.json are both missing")


def _write(payload: dict[str, Any], target: date) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    LATEST_AUDIT_PATH.write_text(text, encoding="utf-8")
    (STATE_DIR / f"postgame_score_audit_{target.isoformat()}.json").write_text(text, encoding="utf-8")


def _int_score(v: Any) -> int:
    if pd.isna(v):
        raise ValueError("score is NA")
    x = float(v)
    if not x.is_integer():
        raise ValueError(f"score is not integer-like: {v}")
    return int(x)


def _is_bottom(v: Any) -> bool:
    s = str(v).strip().lower()
    return s.startswith("bot") or s.startswith("bottom") or s in {"말", "b"}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.update(_flatten(v, key))
            else:
                out[key.lower()] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    return out


def _walk_game_dicts(obj: Any) -> list[dict[str, Any]]:
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


def _pick_score(flat: dict[str, Any], side: str) -> int | None:
    side = side.lower()
    exact_suffixes = (
        f"{side}teamscore",
        f"{side}team.score",
        f"{side}score",
        f"{side}.score",
    )
    candidates: list[Any] = []
    for k, v in flat.items():
        compact = k.replace("_", "")
        if any(compact.endswith(s.replace("_", "")) for s in exact_suffixes):
            candidates.append(v)
    for v in candidates:
        try:
            return _int_score(v)
        except Exception:
            continue
    return None


def fetch_official_final_scores(target: date, expected_ids: list[str]) -> dict[str, dict[str, int]]:
    """Best-effort Naver schedule score cross-check. Absence does not by itself fail the audit."""
    d = target.isoformat()
    params = {
        "fields": "basic,categoryName,statusNum",
        "upperCategoryId": "kbaseball",
        "fromDate": d,
        "toDate": d,
    }
    try:
        r = requests.get(
            NAVER_SCHEDULE_URL,
            params=params,
            headers={"User-Agent": "Mozilla/5.0 KBO-PBP-Audit/1.0"},
            timeout=20,
        )
        r.raise_for_status()
        games = _walk_game_dicts(r.json())
    except Exception:
        return {}

    expected = set(map(str, expected_ids))
    out: dict[str, dict[str, int]] = {}
    for g in games:
        gid = str(g.get("gameId", ""))
        if gid not in expected:
            continue
        flat = _flatten(g)
        home = _pick_score(flat, "home")
        away = _pick_score(flat, "away")
        if home is not None and away is not None:
            out[gid] = {"home": home, "away": away}
    return out


def build_audit(df: pd.DataFrame, run: dict[str, Any], parquet_path: Path) -> dict[str, Any]:
    target = date.fromisoformat(str(run["target_date"]))
    expected_ids = [str(x) for x in (run.get("expected_game_ids") or [])]
    expected_games = int(run.get("expected_games") or 0)
    final_games = int(run.get("final_games") or 0)

    required = {
        "game_pk", "game_date", "home_team", "away_team", "inning", "inning_topbot",
        "at_bat_number", "pitch_number", "post_home_score", "post_away_score",
    }
    missing = sorted(required - set(df.columns))

    dataset_sha = sha256(parquet_path)
    expected_sha = str(run.get("sha256_parquet") or "")
    hash_match = bool(expected_sha) and dataset_sha == expected_sha

    duplicate_stable_keys = None
    if all(k in df.columns for k in STABLE_KEYS):
        duplicate_stable_keys = int(df.duplicated(STABLE_KEYS, keep=False).sum())

    gates = {
        "status_validation_pass": run.get("validation") == "PASS",
        "expected_equals_final": expected_games == final_games,
        "expected_ids_count_match": len(expected_ids) == expected_games,
        "required_columns_present": not missing,
        "dataset_sha256_matches_status": hash_match,
        "stable_key_duplicates_zero": duplicate_stable_keys == 0,
    }

    dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date if "game_date" in df.columns else pd.Series([], dtype=object)
    target_rows = df.loc[dates == target].copy() if len(dates) == len(df) else pd.DataFrame()
    found_ids = sorted(target_rows["game_pk"].astype(str).unique().tolist()) if not target_rows.empty else []
    gates["all_expected_game_ids_present"] = set(expected_ids).issubset(found_ids)

    official = fetch_official_final_scores(target, expected_ids)

    games: list[dict[str, Any]] = []
    if all(gates.values()):
        work = target_rows.copy()
        work["_row_order"] = range(len(work))
        work["_ab"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
        work["_pitch"] = pd.to_numeric(work["pitch_number"], errors="coerce")
        work["_inning"] = pd.to_numeric(work["inning"], errors="coerce")
        work = work.dropna(subset=["game_pk", "_ab", "_inning"]).copy()
        work["_pitch_sort"] = work["_pitch"].fillna(-1)
        work = work.sort_values(
            ["game_pk", "_ab", "_pitch_sort", "_row_order"], kind="stable"
        )
        pa = work.drop_duplicates(["game_pk", "_ab"], keep="last").copy()

        for gid in expected_ids:
            g = pa[pa["game_pk"].astype(str) == gid].sort_values(["_ab", "_row_order"], kind="stable")
            raw_g = target_rows[target_rows["game_pk"].astype(str) == gid]
            row: dict[str, Any] = {"game_pk": gid, "status": "FAIL"}
            problems: list[str] = []

            if g.empty:
                problems.append("no_deduplicated_pa_rows")
                row["problems"] = problems
                games.append(row)
                continue

            away_team = str(g.iloc[0].get("away_team", ""))
            home_team = str(g.iloc[0].get("home_team", ""))
            row["away_team"] = away_team
            row["home_team"] = home_team
            row["game"] = f"{away_team}@{home_team}"
            row["raw_rows"] = int(len(raw_g))
            row["deduplicated_pa_rows"] = int(len(g))

            f5 = g[g["_inning"] == 5]
            if f5.empty:
                problems.append("inning_5_missing")
            else:
                f5_last = f5.iloc[-1]
                if not _is_bottom(f5_last.get("inning_topbot")):
                    problems.append("inning_5_completion_not_identified")
                else:
                    try:
                        row["f5"] = {
                            "away": _int_score(f5_last["post_away_score"]),
                            "home": _int_score(f5_last["post_home_score"]),
                            "last_at_bat_number": int(f5_last["_ab"]),
                        }
                    except Exception as e:
                        problems.append(f"bad_f5_score:{type(e).__name__}")

            final_last = g.iloc[-1]
            try:
                final_away = _int_score(final_last["post_away_score"])
                final_home = _int_score(final_last["post_home_score"])
                row["final"] = {
                    "away": final_away,
                    "home": final_home,
                    "inning": int(final_last["_inning"]),
                    "half": str(final_last.get("inning_topbot", "")),
                    "last_at_bat_number": int(final_last["_ab"]),
                }
            except Exception as e:
                final_away = final_home = None
                problems.append(f"bad_final_score:{type(e).__name__}")

            official_score = official.get(gid)
            if official_score is None:
                row["official_final_crosscheck"] = "NOT_AVAILABLE"
            elif final_away == official_score["away"] and final_home == official_score["home"]:
                row["official_final_crosscheck"] = "PASS"
                row["official_final"] = official_score
            else:
                row["official_final_crosscheck"] = "FAIL"
                row["official_final"] = official_score
                problems.append("official_final_score_mismatch")

            row["problems"] = problems
            row["status"] = "PASS" if not problems else "FAIL"
            games.append(row)

    all_games_pass = len(games) == expected_games and all(g.get("status") == "PASS" for g in games)
    validation = "PASS" if all(gates.values()) and all_games_pass else "FAIL"

    return {
        "version": "1.0.0",
        "validation": validation,
        "target_date": target.isoformat(),
        "generated_at_kst": now_kst().isoformat(),
        "source": "validated_parquet_binary",
        "parquet": parquet_path.name,
        "sha256_parquet": dataset_sha,
        "status_sha256_parquet": expected_sha or None,
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "expected_games": expected_games,
        "final_games": final_games,
        "expected_game_ids": expected_ids,
        "found_game_ids": found_ids,
        "duplicate_stable_keys": duplicate_stable_keys,
        "missing_required_columns": missing,
        "pa_dedup_rule": "one final row per (game_pk, at_bat_number), ordered by pitch_number then source row order",
        "gates": gates,
        "games": games,
        "official_crosscheck_games_found": len(official),
    }


def main() -> int:
    run = _load_run_result()
    target = date.fromisoformat(str(run.get("target_date")))

    if run.get("validation") != "PASS" or int(run.get("expected_games") or 0) == 0:
        payload = {
            "version": "1.0.0",
            "validation": "SKIP",
            "target_date": target.isoformat(),
            "generated_at_kst": now_kst().isoformat(),
            "source": None,
            "reason": "no completed validated target slate available for score audit",
            "updater_state": run.get("state"),
            "updater_validation": run.get("validation"),
            "expected_games": int(run.get("expected_games") or 0),
            "final_games": int(run.get("final_games") or 0),
        }
        _write(payload, target)
        return 0

    parquet_name = str(run.get("parquet") or "")
    parquet_path = OUT_DIR / parquet_name
    if not parquet_name or not parquet_path.exists():
        payload = {
            "version": "1.0.0",
            "validation": "FAIL",
            "target_date": target.isoformat(),
            "generated_at_kst": now_kst().isoformat(),
            "source": "validated_parquet_binary",
            "reason": f"validated parquet not found in out/: {parquet_name or '<missing name>'}",
            "updater_state": run.get("state"),
            "updater_validation": run.get("validation"),
        }
        _write(payload, target)
        return 1

    df = pd.read_parquet(parquet_path)
    payload = build_audit(df, run, parquet_path)
    _write(payload, target)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

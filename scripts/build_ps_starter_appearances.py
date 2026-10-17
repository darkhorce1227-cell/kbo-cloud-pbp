#!/usr/bin/env python3
"""Build reproducible postseason starter appearances with PA-derived actual IP.

The input must be a pitch-level historical postseason master produced by
scripts/build_postseason_master.py. Manual reconstruction rows without pitcher
identity are intentionally unsupported.

Attribution rule for the rare mid-PA pitching change:
    every out recorded by a PA is attributed to the pitcher on the PA's last
    effective row. Such PAs are explicitly flagged in the audit output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "season",
    "game_date",
    "game_pk",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitcher",
    "pitcher_name",
    "outs_when_up",
    "post_outs",
}
PA_KEYS = ["season", "game_pk", "at_bat_number"]


def _clean_id(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def _clean_name(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _half(value) -> str:
    text = str(value).strip().lower()
    aliases = {
        "top": "top",
        "t": "top",
        "초": "top",
        "bot": "bot",
        "bottom": "bot",
        "b": "bot",
        "말": "bot",
    }
    if text not in aliases:
        raise ValueError(f"Unknown inning_topbot value: {value!r}")
    return aliases[text]


def _quality(values: Iterable[object]) -> str:
    cleaned = sorted({str(v).strip() for v in values if not pd.isna(v) and str(v).strip()})
    if not cleaned:
        return "UNAVAILABLE"
    if len(cleaned) == 1:
        return cleaned[0]
    return "MIXED:" + "|".join(cleaned)


def _first_nonempty(values: Iterable[object]) -> str | None:
    for value in values:
        cleaned = _clean_name(value)
        if cleaned:
            return cleaned
    return None


def build_starter_appearances(data: pd.DataFrame, source_dataset: str):
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            "Input is not a pitcher-identifiable pitch-level PS master; "
            f"missing columns: {', '.join(missing)}. "
            "Manual reconstruction without pitcher identity must remain UNAVAILABLE."
        )
    if data.empty:
        raise ValueError("Input postseason master is empty")

    df = data.copy().reset_index(drop=True)
    df["_source_row"] = range(len(df))
    df["pitcher_id_clean"] = df["pitcher"].map(_clean_id)
    df["pitcher_name_clean"] = df["pitcher_name"].map(_clean_name)
    df["half"] = df["inning_topbot"].map(_half)

    for col in ("season", "inning", "at_bat_number", "outs_when_up", "post_outs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["season", "game_pk", "inning", "at_bat_number", "outs_when_up", "post_outs"]].isna().any().any():
        bad = df[
            df[["season", "game_pk", "inning", "at_bat_number", "outs_when_up", "post_outs"]]
            .isna()
            .any(axis=1)
        ]
        raise ValueError(f"Missing/non-numeric essential state in {len(bad)} pitch rows")

    df["season"] = df["season"].astype(int)
    df["inning"] = df["inning"].astype(int)
    df["at_bat_number"] = df["at_bat_number"].astype(int)
    df["outs_when_up"] = df["outs_when_up"].astype(int)
    df["post_outs"] = df["post_outs"].astype(int)

    # A PA is parser-defined by game + at_bat_number. All stamped PA state must
    # be constant across its pitch rows. Preserve source order and select the
    # final effective row for pitcher attribution.
    pa_group = df.groupby(PA_KEYS, sort=False, dropna=False)
    state_nunique = pa_group[["inning", "half", "outs_when_up", "post_outs"]].nunique(dropna=False)
    inconsistent_state_mask = (state_nunique > 1).any(axis=1)
    inconsistent_pa_keys = set(state_nunique.index[inconsistent_state_mask].tolist())

    pitcher_nunique = pa_group["pitcher_id_clean"].nunique(dropna=True)
    mid_pa_keys = set(pitcher_nunique.index[pitcher_nunique > 1].tolist())

    pa = pa_group.tail(1).copy()
    pa["actual_outs"] = pa["post_outs"] - pa["outs_when_up"]
    pa["mid_pa_pitcher_change"] = [
        tuple(row) in mid_pa_keys for row in pa[PA_KEYS].itertuples(index=False, name=None)
    ]
    pa["state_inconsistent"] = [
        tuple(row) in inconsistent_pa_keys for row in pa[PA_KEYS].itertuples(index=False, name=None)
    ]
    pa["attributed_pitcher_id"] = pa["pitcher_id_clean"]
    pa["attributed_pitcher_name"] = pa["pitcher_name_clean"]

    bad_delta = pa[(pa["actual_outs"] < 0) | (pa["actual_outs"] > 3)]
    missing_attr = pa[pa["attributed_pitcher_id"].isna() & (pa["actual_outs"] > 0)]

    # TOP = away bats = home fields; BOT = home bats = away fields.
    pa["fielding_team"] = pa.apply(
        lambda r: r["home_team"] if r["half"] == "top" else r["away_team"], axis=1
    )
    pa["batting_team"] = pa.apply(
        lambda r: r["away_team"] if r["half"] == "top" else r["home_team"], axis=1
    )
    df["fielding_team"] = df.apply(
        lambda r: r["home_team"] if r["half"] == "top" else r["away_team"], axis=1
    )
    df["batting_team"] = df.apply(
        lambda r: r["away_team"] if r["half"] == "top" else r["home_team"], axis=1
    )

    # Starter = first pitcher to appear while that team is fielding. Use every
    # source row, so starter identification is independent of outs attribution.
    first_pitcher_rows = (
        df[df["pitcher_id_clean"].notna()]
        .sort_values("_source_row")
        .groupby(["season", "game_pk", "fielding_team"], sort=False, as_index=False)
        .first()
    )

    name_lookup = (
        df[df["pitcher_id_clean"].notna()]
        .groupby(["season", "game_pk", "pitcher_id_clean"], sort=False)["pitcher_name_clean"]
        .agg(_first_nonempty)
        .to_dict()
    )

    all_pitcher_outs = (
        pa[pa["attributed_pitcher_id"].notna()]
        .groupby(["season", "game_pk", "fielding_team", "attributed_pitcher_id"], as_index=False)
        .agg(actual_outs=("actual_outs", "sum"), batters_faced=("at_bat_number", "count"))
    )
    outs_lookup = {
        (int(r.season), r.game_pk, r.fielding_team, r.attributed_pitcher_id):
        (int(r.actual_outs), int(r.batters_faced))
        for r in all_pitcher_outs.itertuples(index=False)
    }

    appearance = (
        df[df["pitcher_id_clean"].notna()]
        .groupby(["season", "game_pk", "fielding_team", "pitcher_id_clean"], as_index=False)
        .agg(first_inning=("inning", "min"), last_inning=("inning", "max"))
    )
    appearance_lookup = {
        (int(r.season), r.game_pk, r.fielding_team, r.pitcher_id_clean):
        (int(r.first_inning), int(r.last_inning))
        for r in appearance.itertuples(index=False)
    }

    quality_col = "data_quality" if "data_quality" in df.columns else None
    rows = []
    for r in first_pitcher_rows.itertuples(index=False):
        season = int(r.season)
        game_pk = r.game_pk
        team = r.fielding_team
        starter_id = r.pitcher_id_clean
        home_away = "home" if team == r.home_team else "away"
        opponent = r.away_team if home_away == "home" else r.home_team
        actual_outs, bf = outs_lookup.get((season, game_pk, team, starter_id), (0, 0))
        first_inning, last_inning = appearance_lookup[(season, game_pk, team, starter_id)]
        starter_name = name_lookup.get((season, game_pk, starter_id)) or r.pitcher_name_clean
        game_mask = (df["season"] == season) & (df["game_pk"] == game_pk)
        source_quality = _quality(df.loc[game_mask, quality_col]) if quality_col else "UNAVAILABLE"

        starter_mid_pa_involved = False
        for key in mid_pa_keys:
            if int(key[0]) != season or key[1] != game_pk:
                continue
            pa_rows = df[
                (df["season"] == season)
                & (df["game_pk"] == game_pk)
                & (df["at_bat_number"] == int(key[2]))
            ]
            if (pa_rows["pitcher_id_clean"] == starter_id).any():
                starter_mid_pa_involved = True
                break

        flags = []
        if starter_mid_pa_involved:
            flags.append("MID_PA_CHANGE_INVOLVED_FINAL_PITCHER_ATTRIBUTION")
        game_pa_mask = (pa["season"] == season) & (pa["game_pk"] == game_pk)
        if pa.loc[game_pa_mask, "state_inconsistent"].any():
            flags.append("GAME_HAS_INCONSISTENT_PA_STATE")
        if first_inning != 1:
            flags.append("STARTER_FIRST_APPEARANCE_NOT_INNING_1")

        rows.append({
            "season": season,
            "game_date": r.game_date,
            "game_pk": game_pk,
            "team": team,
            "opponent": opponent,
            "home_away": home_away,
            "starter_id": starter_id,
            "starter_name": starter_name,
            "actual_outs": actual_outs,
            "actual_ip": actual_outs / 3.0,
            "starter_first_inning": first_inning,
            "starter_last_inning": last_inning,
            "batters_faced": bf,
            "source_quality": source_quality,
            "source_dataset": source_dataset,
            "attribution_integrity_flag": "PASS" if not flags else "|".join(flags),
        })

    starters = pd.DataFrame(rows).sort_values(
        ["season", "game_date", "game_pk", "home_away"], kind="stable"
    ).reset_index(drop=True)

    starters_per_game = starters.groupby(["season", "game_pk"]).size()
    bad_starter_games = starters_per_game[starters_per_game != 2]

    # Compare all PA-derived defensive outs with outs successfully attributed to
    # a pitcher. This is not an external boxscore check; it detects attribution
    # loss or missing pitcher identity inside the pitch master.
    defense_outs = (
        pa.groupby(["season", "game_pk", "fielding_team"], as_index=False)["actual_outs"]
        .sum()
        .rename(columns={"actual_outs": "pa_defense_outs"})
    )
    pitcher_outs = (
        all_pitcher_outs.groupby(["season", "game_pk", "fielding_team"], as_index=False)["actual_outs"]
        .sum()
        .rename(columns={"actual_outs": "pitcher_attributed_outs"})
    )
    defense_audit = defense_outs.merge(
        pitcher_outs, on=["season", "game_pk", "fielding_team"], how="left"
    )
    defense_audit["pitcher_attributed_outs"] = (
        defense_audit["pitcher_attributed_outs"].fillna(0).astype(int)
    )
    defense_audit["outs_match"] = (
        defense_audit["pa_defense_outs"].astype(int)
        == defense_audit["pitcher_attributed_outs"]
    )
    bad_defense = defense_audit[~defense_audit["outs_match"]]

    thirds_ok = ((starters["actual_ip"] * 3 - starters["actual_outs"]).abs() < 1e-12).all()
    nonnegative_ok = (starters["actual_outs"] >= 0).all()
    pa_dedupe_ok = len(pa) == len(df[PA_KEYS].drop_duplicates())

    critical_failures = {
        "games_with_starter_count_ne_2": int(len(bad_starter_games)),
        "pa_state_inconsistent": int(len(inconsistent_pa_keys)),
        "pa_out_delta_invalid": int(len(bad_delta)),
        "positive_out_pa_missing_pitcher": int(len(missing_attr)),
        "defense_outs_mismatch": int(len(bad_defense)),
        "starter_outs_negative": int(not nonnegative_ok),
        "actual_ip_not_thirds": int(not thirds_ok),
        "pa_dedupe_failed": int(not pa_dedupe_ok),
    }
    status = "PASS" if not any(critical_failures.values()) else "FAIL"

    audit = {
        "status": status,
        "source_dataset": source_dataset,
        "pitch_rows": int(len(df)),
        "games": int(df[["season", "game_pk"]].drop_duplicates().shape[0]),
        "plate_appearances": int(len(pa)),
        "starter_appearances": int(len(starters)),
        "mid_pa_pitcher_change_pas": int(len(mid_pa_keys)),
        "attribution_rule": "PA outs -> pitcher on final effective row of PA",
        "manual_reconstruction_policy": (
            "UNAVAILABLE unless pitcher identity exists; never merged by inference"
        ),
        "batters_faced_definition": (
            "count of PAs attributed to pitcher under final-effective-row rule; "
            "not claimed as official BF for mid-PA changes"
        ),
        "checks": {
            "exactly_two_starters_per_game": int(len(bad_starter_games)) == 0,
            "starter_actual_outs_nonnegative": bool(nonnegative_ok),
            "actual_ip_is_outs_div_3": bool(thirds_ok),
            "pa_dedupe_one_row_per_game_at_bat_number": bool(pa_dedupe_ok),
            "pa_state_constant_within_pa": int(len(inconsistent_pa_keys)) == 0,
            "mid_pa_pitcher_changes_detected_and_flagged": True,
            "all_positive_out_pas_have_pitcher_identity": int(len(missing_attr)) == 0,
            "pitcher_outs_equal_pa_derived_defense_outs": int(len(bad_defense)) == 0,
        },
        "critical_failures": critical_failures,
        "bad_starter_games": [
            {"season": int(idx[0]), "game_pk": idx[1], "starter_count": int(value)}
            for idx, value in bad_starter_games.items()
        ],
        "bad_defense_outs": defense_audit.loc[
            ~defense_audit["outs_match"],
            ["season", "game_pk", "fielding_team", "pa_defense_outs", "pitcher_attributed_outs"],
        ].to_dict("records"),
    }
    return starters, audit, defense_audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="historical_ps/postseason_all_available.parquet",
        help="Pitch-level postseason master parquet",
    )
    ap.add_argument(
        "--output",
        default="historical_ps/ps_starter_appearances.csv",
        help="Starter appearance CSV",
    )
    ap.add_argument(
        "--audit-output",
        default="historical_ps/ps_starter_appearances_validation.json",
        help="Validation report JSON",
    )
    ap.add_argument(
        "--defense-audit-output",
        default="historical_ps/ps_starter_appearances_defense_outs_audit.csv",
        help="Game/team PA-outs vs pitcher-attributed-outs audit CSV",
    )
    ap.add_argument(
        "--allow-audit-fail",
        action="store_true",
        help="Write outputs but exit 0 even when critical validation fails",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() != ".parquet":
        raise ValueError("Expected a .parquet pitch-level postseason master")
    data = pd.read_parquet(input_path)
    starters, audit, defense_audit = build_starter_appearances(data, input_path.name)

    output = Path(args.output)
    audit_output = Path(args.audit_output)
    defense_output = Path(args.defense_audit_output)
    for path in (output, audit_output, defense_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    starters.to_csv(output, index=False)
    defense_audit.to_csv(defense_output, index=False)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "games": audit["games"],
                "starter_appearances": audit["starter_appearances"],
                "mid_pa_pitcher_change_pas": audit["mid_pa_pitcher_change_pas"],
                "output": str(output),
                "audit": str(audit_output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if audit["status"] != "PASS" and not args.allow_audit_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

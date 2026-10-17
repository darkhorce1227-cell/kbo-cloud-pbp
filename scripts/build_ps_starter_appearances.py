#!/usr/bin/env python3
"""Build reproducible postseason starter appearances with PA-derived actual IP.

Production rows are emitted only when the pitch-level source has auditable
half-inning out-state semantics. Games with incomplete coverage, broken out
state, or missing pitcher identity are kept out of the production table and
reported as UNAVAILABLE in the validation JSON.

Attribution rule for the rare mid-PA pitching change:
    every out recorded by a PA is attributed to the pitcher on the PA's final
    effective row. Such PAs are explicitly flagged for audit.

Manual reconstruction rows without pitcher identity are intentionally not
merged or inferred.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    "post_home_score",
    "post_away_score",
}
PA_KEYS = ["season", "game_pk", "at_bat_number"]
GAME_KEYS = ["season", "game_pk"]
HALF_KEYS = ["season", "game_pk", "inning", "half"]


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


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    return set(frame[columns].itertuples(index=False, name=None))


def _subset_by_game_keys(frame: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    if not keys:
        return frame.iloc[0:0].copy()
    mask = [tuple(row) in keys for row in frame[GAME_KEYS].itertuples(index=False, name=None)]
    return frame.loc[mask].copy()


def _audit_game_out_state(pa: pd.DataFrame) -> dict[tuple, set[str]]:
    """Return game-level reasons that make actual IP unavailable.

    A trustworthy game must have complete inning/half coverage and internally
    coherent out-state transitions. Every non-final half must end at three outs.
    A final bottom half may end before three outs only on a walk-off, verified by
    the stamped post-play score.
    """
    reasons: dict[tuple, set[str]] = defaultdict(set)

    # PA-local state errors are source failures for the whole game.
    bad_delta = pa[(pa["actual_outs"] < 0) | (pa["actual_outs"] > 3)]
    for season, game_pk in bad_delta[GAME_KEYS].drop_duplicates().itertuples(index=False, name=None):
        reasons[(int(season), game_pk)].add("INVALID_PA_OUT_DELTA")

    bad_state = pa[pa["state_inconsistent"]]
    for season, game_pk in bad_state[GAME_KEYS].drop_duplicates().itertuples(index=False, name=None):
        reasons[(int(season), game_pk)].add("PA_STATE_INCONSISTENT")

    missing_pitcher = pa[pa["attributed_pitcher_id"].isna()]
    for season, game_pk in missing_pitcher[GAME_KEYS].drop_duplicates().itertuples(index=False, name=None):
        reasons[(int(season), game_pk)].add("PA_PITCHER_ID_UNAVAILABLE")

    # Coverage: every inning must have a top half; every inning before the last
    # must also have a bottom half. The final bottom can be absent when the home
    # team already leads after the top half.
    for (season, game_pk), g in pa.groupby(GAME_KEYS, sort=False):
        season = int(season)
        game_key = (season, game_pk)
        innings = sorted(int(v) for v in g["inning"].unique())
        if not innings:
            reasons[game_key].add("NO_INNINGS")
            continue
        max_inning = max(innings)
        expected_innings = list(range(1, max_inning + 1))
        if innings != expected_innings:
            reasons[game_key].add("MISSING_INNING")
        for inning in expected_innings:
            halves = set(g.loc[g["inning"] == inning, "half"].tolist())
            if "top" not in halves:
                reasons[game_key].add("MISSING_TOP_HALF")
            if inning < max_inning and "bot" not in halves:
                reasons[game_key].add("MISSING_BOTTOM_BEFORE_FINAL_INNING")

    # Half-inning semantic audit.
    half_rows = []
    for (season, game_pk, inning, half), h in pa.groupby(HALF_KEYS, sort=False):
        h = h.sort_values("_source_row")
        first = h.iloc[0]
        last = h.iloc[-1]
        half_rows.append(
            {
                "season": int(season),
                "game_pk": game_pk,
                "inning": int(inning),
                "half": half,
                "first_outs": int(first["outs_when_up"]),
                "final_outs": int(last["post_outs"]),
                "sum_delta": int(h["actual_outs"].sum()),
                "last_source_row": int(last["_source_row"]),
                "post_home_score": int(last["post_home_score"]),
                "post_away_score": int(last["post_away_score"]),
            }
        )
    halves = pd.DataFrame(half_rows)

    if not halves.empty:
        last_half_idx = halves.groupby(GAME_KEYS, sort=False)["last_source_row"].idxmax()
        last_half_keys = set(
            halves.loc[last_half_idx, ["season", "game_pk", "inning", "half"]]
            .itertuples(index=False, name=None)
        )

        for r in halves.itertuples(index=False):
            game_key = (int(r.season), r.game_pk)
            half_key = (int(r.season), r.game_pk, int(r.inning), r.half)
            is_last_half = half_key in last_half_keys

            if r.first_outs != 0:
                reasons[game_key].add("HALF_START_OUTS_NOT_ZERO")
            if r.sum_delta != r.final_outs:
                reasons[game_key].add("HALF_OUT_DELTA_NOT_FINAL_OUTS")

            if not is_last_half:
                if r.final_outs != 3:
                    reasons[game_key].add("NONFINAL_HALF_DID_NOT_END_AT_3_OUTS")
            elif r.half == "top":
                if r.final_outs != 3:
                    reasons[game_key].add("FINAL_TOP_DID_NOT_END_AT_3_OUTS")
            else:
                if r.final_outs < 3 and not (r.post_home_score > r.post_away_score):
                    reasons[game_key].add("FINAL_BOTTOM_LT3_NOT_WALKOFF")
                if r.final_outs > 3:
                    reasons[game_key].add("FINAL_BOTTOM_OUTS_GT3")

    return reasons


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

    numeric_cols = (
        "season",
        "inning",
        "at_bat_number",
        "outs_when_up",
        "post_outs",
        "post_home_score",
        "post_away_score",
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    essential = ["season", "game_pk", "inning", "at_bat_number", "outs_when_up", "post_outs", "post_home_score", "post_away_score"]
    if df[essential].isna().any().any():
        bad = df[df[essential].isna().any(axis=1)]
        raise ValueError(f"Missing/non-numeric essential state in {len(bad)} pitch rows")

    for col in numeric_cols:
        df[col] = df[col].astype(int)

    input_game_keys = _key_set(df, GAME_KEYS)

    # A PA is parser-defined by game + at_bat_number. All stamped PA state must
    # be constant across its pitch rows. Preserve source order and select the
    # final effective row for pitcher attribution.
    pa_group = df.groupby(PA_KEYS, sort=False, dropna=False)
    state_nunique = pa_group[["inning", "half", "outs_when_up", "post_outs"]].nunique(dropna=False)
    inconsistent_state_mask = (state_nunique > 1).any(axis=1)
    inconsistent_pa_keys = set(state_nunique.index[inconsistent_state_mask].tolist())

    pitcher_nunique = pa_group["pitcher_id_clean"].nunique(dropna=True)
    mid_pa_keys = set(pitcher_nunique.index[pitcher_nunique > 1].tolist())

    pa = pa_group.tail(1).copy().sort_values("_source_row", kind="stable")
    pa["actual_outs"] = pa["post_outs"] - pa["outs_when_up"]
    pa["mid_pa_pitcher_change"] = [
        tuple(row) in mid_pa_keys for row in pa[PA_KEYS].itertuples(index=False, name=None)
    ]
    pa["state_inconsistent"] = [
        tuple(row) in inconsistent_pa_keys for row in pa[PA_KEYS].itertuples(index=False, name=None)
    ]
    pa["attributed_pitcher_id"] = pa["pitcher_id_clean"]
    pa["attributed_pitcher_name"] = pa["pitcher_name_clean"]

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

    unavailable_reasons = _audit_game_out_state(pa)
    candidate_game_keys = input_game_keys - set(unavailable_reasons)

    # Starter = first pitcher to appear while that team is fielding. Do this on
    # semantic-PASS games, then exclude any game that still cannot identify two
    # starters exactly.
    candidate_df = _subset_by_game_keys(df, candidate_game_keys)
    candidate_first = (
        candidate_df[candidate_df["pitcher_id_clean"].notna()]
        .sort_values("_source_row", kind="stable")
        .groupby(["season", "game_pk", "fielding_team"], sort=False, group_keys=False)
        .head(1)
    )
    starter_counts = candidate_first.groupby(GAME_KEYS).size().to_dict()
    for game_key in candidate_game_keys:
        count = int(starter_counts.get(game_key, 0))
        if count != 2:
            unavailable_reasons[game_key].add(f"STARTER_COUNT_{count}_NE_2")

    production_game_keys = input_game_keys - set(unavailable_reasons)
    valid_df = _subset_by_game_keys(df, production_game_keys)
    valid_pa = _subset_by_game_keys(pa, production_game_keys)

    first_pitcher_rows = (
        valid_df[valid_df["pitcher_id_clean"].notna()]
        .sort_values("_source_row", kind="stable")
        .groupby(["season", "game_pk", "fielding_team"], sort=False, group_keys=False)
        .head(1)
    )

    name_lookup = (
        valid_df[valid_df["pitcher_id_clean"].notna()]
        .groupby(["season", "game_pk", "pitcher_id_clean"], sort=False)["pitcher_name_clean"]
        .agg(_first_nonempty)
        .to_dict()
    )

    all_pitcher_outs = (
        valid_pa[valid_pa["attributed_pitcher_id"].notna()]
        .groupby(["season", "game_pk", "fielding_team", "attributed_pitcher_id"], as_index=False)
        .agg(actual_outs=("actual_outs", "sum"), batters_faced=("at_bat_number", "count"))
    )
    outs_lookup = {
        (int(r.season), r.game_pk, r.fielding_team, r.attributed_pitcher_id):
        (int(r.actual_outs), int(r.batters_faced))
        for r in all_pitcher_outs.itertuples(index=False)
    }

    appearance = (
        valid_df[valid_df["pitcher_id_clean"].notna()]
        .groupby(["season", "game_pk", "fielding_team", "pitcher_id_clean"], as_index=False)
        .agg(first_inning=("inning", "min"), last_inning=("inning", "max"))
    )
    appearance_lookup = {
        (int(r.season), r.game_pk, r.fielding_team, r.pitcher_id_clean):
        (int(r.first_inning), int(r.last_inning))
        for r in appearance.itertuples(index=False)
    }

    quality_col = "data_quality" if "data_quality" in valid_df.columns else None
    rows = []
    for r in first_pitcher_rows.itertuples(index=False):
        season = int(r.season)
        game_pk = r.game_pk
        team = r.fielding_team
        starter_id = r.pitcher_id_clean
        home_away = "home" if team == r.home_team else "away"
        opponent = r.away_team if home_away == "home" else r.home_team
        actual_outs, bf = outs_lookup[(season, game_pk, team, starter_id)]
        first_inning, last_inning = appearance_lookup[(season, game_pk, team, starter_id)]
        starter_name = name_lookup.get((season, game_pk, starter_id)) or r.pitcher_name_clean
        game_mask = (valid_df["season"] == season) & (valid_df["game_pk"] == game_pk)
        source_quality = _quality(valid_df.loc[game_mask, quality_col]) if quality_col else "UNAVAILABLE"

        starter_mid_pa_involved = False
        for key in mid_pa_keys:
            if int(key[0]) != season or key[1] != game_pk:
                continue
            pa_rows = valid_df[
                (valid_df["season"] == season)
                & (valid_df["game_pk"] == game_pk)
                & (valid_df["at_bat_number"] == int(key[2]))
            ]
            if (pa_rows["pitcher_id_clean"] == starter_id).any():
                starter_mid_pa_involved = True
                break

        flags = []
        if starter_mid_pa_involved:
            flags.append("MID_PA_CHANGE_INVOLVED_FINAL_PITCHER_ATTRIBUTION")
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
            "out_state_integrity": "PASS",
            "attribution_integrity_flag": "PASS" if not flags else "|".join(flags),
        })

    starters = pd.DataFrame(rows)
    if not starters.empty:
        starters = starters.sort_values(
            ["season", "game_date", "game_pk", "home_away"], kind="stable"
        ).reset_index(drop=True)

    starters_per_game = starters.groupby(GAME_KEYS).size() if not starters.empty else pd.Series(dtype=int)
    bad_starter_games = starters_per_game[starters_per_game != 2]

    # Compare PA-derived defensive outs with outs successfully attributed to a
    # pitcher inside production games.
    defense_outs = (
        valid_pa.groupby(["season", "game_pk", "fielding_team"], as_index=False)["actual_outs"]
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
    if not defense_audit.empty:
        defense_audit["pitcher_attributed_outs"] = defense_audit["pitcher_attributed_outs"].fillna(0).astype(int)
        defense_audit["outs_match"] = (
            defense_audit["pa_defense_outs"].astype(int)
            == defense_audit["pitcher_attributed_outs"]
        )
    else:
        defense_audit["outs_match"] = pd.Series(dtype=bool)
    bad_defense = defense_audit[~defense_audit["outs_match"]]

    thirds_ok = bool(
        starters.empty
        or ((starters["actual_ip"] * 3 - starters["actual_outs"]).abs() < 1e-12).all()
    )
    nonnegative_ok = bool(starters.empty or (starters["actual_outs"] >= 0).all())
    pa_dedupe_ok = len(pa) == len(df[PA_KEYS].drop_duplicates())
    production_starter_count_ok = bool(
        len(production_game_keys) > 0
        and len(starters) == 2 * len(production_game_keys)
        and len(bad_starter_games) == 0
    )

    critical_failures = {
        "production_games_with_starter_count_ne_2": int(len(bad_starter_games)),
        "production_defense_outs_mismatch": int(len(bad_defense)),
        "starter_outs_negative": int(not nonnegative_ok),
        "actual_ip_not_thirds": int(not thirds_ok),
        "pa_dedupe_failed": int(not pa_dedupe_ok),
        "no_production_games": int(len(production_game_keys) == 0),
        "production_starter_count_failed": int(not production_starter_count_ok),
    }
    status = "PASS" if not any(critical_failures.values()) else "FAIL"

    unavailable_rows = [
        {
            "season": int(season),
            "game_pk": game_pk,
            "status": "UNAVAILABLE",
            "reasons": sorted(reasons),
        }
        for (season, game_pk), reasons in sorted(unavailable_reasons.items(), key=lambda x: (int(x[0][0]), str(x[0][1])))
    ]

    availability_by_season = []
    input_games_df = df[GAME_KEYS].drop_duplicates()
    prod_games_df = valid_df[GAME_KEYS].drop_duplicates()
    seasons = sorted(int(v) for v in input_games_df["season"].unique())
    for season in seasons:
        input_n = int((input_games_df["season"] == season).sum())
        prod_n = int((prod_games_df["season"] == season).sum())
        availability_by_season.append(
            {
                "season": season,
                "input_games": input_n,
                "production_games": prod_n,
                "unavailable_games": input_n - prod_n,
            }
        )

    def coverage(lo: int, hi: int) -> dict:
        g = prod_games_df[prod_games_df["season"].between(lo, hi)]
        s = starters[starters["season"].between(lo, hi)] if not starters.empty else starters
        return {
            "seasons": f"{lo}-{hi}",
            "games": int(len(g)),
            "starter_appearances": int(len(s)),
        }

    production_mid_pa_keys = {
        key for key in mid_pa_keys if (int(key[0]), key[1]) in production_game_keys
    }

    audit = {
        "status": status,
        "source_dataset": source_dataset,
        "input_pitch_rows": int(len(df)),
        "input_games": int(len(input_game_keys)),
        "input_plate_appearances": int(len(pa)),
        "production_pitch_rows": int(len(valid_df)),
        "production_games": int(len(production_game_keys)),
        "production_plate_appearances": int(len(valid_pa)),
        "starter_appearances": int(len(starters)),
        "unavailable_games": int(len(unavailable_reasons)),
        "mid_pa_pitcher_change_pas_input": int(len(mid_pa_keys)),
        "mid_pa_pitcher_change_pas_production": int(len(production_mid_pa_keys)),
        "attribution_rule": "PA outs -> pitcher on final effective row of PA",
        "out_state_policy": (
            "Only games with complete half-inning coverage and coherent out-state semantics "
            "are emitted; all others remain UNAVAILABLE."
        ),
        "manual_reconstruction_policy": (
            "UNAVAILABLE unless pitcher identity exists; never merged by inference"
        ),
        "batters_faced_definition": (
            "count of PAs attributed to pitcher under final-effective-row rule; "
            "not claimed as official BF for mid-PA changes"
        ),
        "coverage": {
            "2016_2025": coverage(2016, 2025),
            "2017_2025": coverage(2017, 2025),
        },
        "availability_by_season": availability_by_season,
        "checks": {
            "production_games_have_exactly_two_starters": production_starter_count_ok,
            "starter_actual_outs_nonnegative": nonnegative_ok,
            "actual_ip_is_outs_div_3": thirds_ok,
            "pa_dedupe_one_row_per_game_at_bat_number": pa_dedupe_ok,
            "unavailable_source_games_excluded_not_inferred": True,
            "mid_pa_pitcher_changes_detected_and_flagged": True,
            "pitcher_outs_equal_pa_derived_defense_outs": int(len(bad_defense)) == 0,
        },
        "critical_failures": critical_failures,
        "unavailable_game_details": unavailable_rows,
        "bad_production_starter_games": [
            {"season": int(idx[0]), "game_pk": idx[1], "starter_count": int(value)}
            for idx, value in bad_starter_games.items()
        ],
        "bad_production_defense_outs": defense_audit.loc[
            ~defense_audit["outs_match"],
            ["season", "game_pk", "fielding_team", "pa_defense_outs", "pitcher_attributed_outs"],
        ].to_dict("records"),
    }
    return starters, audit, defense_audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="historical_ps/postseason_screen_pass.parquet",
        help="Pitch-level postseason master parquet",
    )
    ap.add_argument(
        "--output",
        default="historical_ps/ps_starter_appearances.csv",
        help="Production starter appearance CSV; source-unavailable games are excluded",
    )
    ap.add_argument(
        "--audit-output",
        default="historical_ps/ps_starter_appearances_validation.json",
        help="Validation report JSON including UNAVAILABLE game details",
    )
    ap.add_argument(
        "--defense-audit-output",
        default="historical_ps/ps_starter_appearances_defense_outs_audit.csv",
        help="Production game/team PA-outs vs pitcher-attributed-outs audit CSV",
    )
    ap.add_argument(
        "--allow-audit-fail",
        action="store_true",
        help="Write outputs but exit 0 even when production validation fails",
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
                "input_games": audit["input_games"],
                "production_games": audit["production_games"],
                "unavailable_games": audit["unavailable_games"],
                "starter_appearances": audit["starter_appearances"],
                "coverage_2017_2025": audit["coverage"]["2017_2025"],
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

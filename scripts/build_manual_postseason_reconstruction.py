#!/usr/bin/env python3
"""Build an audited effective PA view for manually reconstructed missing postseason games.

The manual source files intentionally preserve provenance. This script unions the
base PA reconstruction with later source-backed additions, applies the small
correction overlay, and runs conservative sanity checks. It does not synthesize
missing plate appearances or pitches.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

BLANK_SENTINEL = "__BLANK__"
KEY_FIELDS = ("game_id", "inning", "half", "sequence_order")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def apply_corrections(rows: list[dict[str, str]], corrections: list[dict[str, str]]) -> int:
    by_key = {tuple(r[k] for k in KEY_FIELDS): r for r in rows}
    applied = 0
    for c in corrections:
        key = tuple(c[k] for k in KEY_FIELDS)
        if key not in by_key:
            raise ValueError(f"correction target not found: {key}")
        field = c["field"]
        if field not in by_key[key]:
            raise ValueError(f"correction field not found: {field}")
        value = c["override_value"]
        by_key[key][field] = "" if value == BLANK_SENTINEL else value
        applied += 1
    return applied


def validate(rows: list[dict[str, str]]) -> dict[str, object]:
    keys = [tuple(r[k] for k in KEY_FIELDS) for r in rows]
    duplicates = [k for k, n in Counter(keys).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate manual PA keys: {duplicates[:5]}")

    counts = Counter(r["game_id"] for r in rows)
    expected = {
        "33331008SSLT0": 84,
        "33331009SSLT0": 20,
        "33331011LTSS0": 24,
        "33331011SKHT0": 12,
        "77771026SKSS0": 20,
        "33331011OBLT0": 15,
        "77771029SSOB0": 18,
    }
    if dict(counts) != expected:
        raise ValueError(f"unexpected per-game row counts: {dict(counts)}")

    # The only full-game PA reconstruction currently available is 2008 QP Game 1.
    g1 = [r for r in rows if r["game_id"] == "33331008SSLT0"]
    if not all(r["coverage_scope"] == "COMPLETE_GAME_PA_ORDER" for r in g1):
        raise ValueError("2008 QP Game 1 contains a non-complete coverage row")
    run_totals = defaultdict(int)
    for r in g1:
        if r["runs_scored"].isdigit():
            run_totals[r["batting_team"]] += int(r["runs_scored"])
    if dict(run_totals) != {"삼성": 12, "롯데": 3}:
        raise ValueError(f"2008 QP Game 1 run audit failed: {dict(run_totals)}")

    # These 2013 fields are intentionally unknown after the audit. Blank is
    # materially different from zero and must stay blank until a stronger source appears.
    unknown_2013 = {
        ("77771029SSOB0", "1", "TOP", str(i)) for i in range(2, 6)
    } | {("77771029SSOB0", "3", "BOT", "3")}
    by_key = {tuple(r[k] for k in KEY_FIELDS): r for r in rows}
    for key in unknown_2013:
        if by_key[key]["runs_scored"] != "":
            raise ValueError(f"ambiguous 2013 run attribution must be blank: {key}")

    # The recovered 2008 QP Game 3 fourth-inning scoring row must survive the union.
    g3_rbi = by_key[("33331011LTSS0", "4", "TOP", "2")]
    if g3_rbi["batter"] != "강민호" or g3_rbi["runs_scored"] != "1":
        raise ValueError("2008 QP Game 3 fourth-inning recovery audit failed")

    # Additional 2008 photo-report anchors from the third audit pass.
    g2_hbp = by_key[("33331009SSLT0", "6", "TOP", "1")]
    if g2_hbp["batter"] != "채태인" or g2_hbp["pa_result"] != "HBP":
        raise ValueError("2008 QP Game 2 sixth-inning HBP audit failed")
    g3_home_out = by_key[("33331011LTSS0", "5", "TOP", "3")]
    if g3_home_out["batter"] != "이대호" or g3_home_out["pa_result"] != "GROUNDOUT_RUNNER_OUT_HOME":
        raise ValueError("2008 QP Game 3 fifth-inning home-out audit failed")

    # 2011 KS Game 2 first inning is independently described from first batter to third out.
    ks2_first = [
        r for r in rows
        if r["game_id"] == "77771026SKSS0" and r["inning"] == "1" and r["half"] == "TOP"
    ]
    if len(ks2_first) != 5 or not all(
        r["coverage_scope"] == "COMPLETE_HALF_INNING_PA_ORDER" and r["order_semantics"] == "COMPLETE_PA_ORDER"
        for r in ks2_first
    ):
        raise ValueError("2011 KS Game 2 first-inning complete-order audit failed")

    return {
        "rows": len(rows),
        "games": len(counts),
        "per_game_rows": dict(counts),
        "complete_game_pa_rows": len(g1),
        "complete_game_run_audit": {"삼성": 12, "롯데": 3},
        "ks2_complete_first_inning_rows": len(ks2_first),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="historical_ps/manual_reconstruction")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    source = root / "pa_reconstruction.csv"
    additions_file = root / "pa_reconstruction_additions.csv"
    correction_file = root / "pa_reconstruction_corrections.csv"
    output = Path(args.output) if args.output else root / "pa_reconstruction_effective.csv"

    rows = read_rows(source)
    fields = list(rows[0].keys())
    additions = read_rows(additions_file) if additions_file.exists() else []
    for r in additions:
        if list(r.keys()) != fields:
            raise ValueError("PA addition schema differs from base reconstruction")
    rows.extend(additions)

    corrections = read_rows(correction_file)
    applied = apply_corrections(rows, corrections)
    summary = validate(rows)
    write_rows(output, rows, fields)

    print({
        "output": str(output),
        "additions": len(additions),
        "corrections_applied": applied,
        **summary,
    })


if __name__ == "__main__":
    main()

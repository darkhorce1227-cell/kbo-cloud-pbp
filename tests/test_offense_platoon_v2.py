"""Meaningful model-integrity regressions; run with the standard library."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_kbo_pitcher_hands import parse_profile
from research_offense_platoon_v2 import (
    BASE_FEATURES, DELTA_FEATURES, PLATOON_FEATURES, PLAN_PATH,
    build_rows, phase_distributions, wdl_from_pmf, prepare_games,
)


def synthetic_game(pk, date, home="B", away="A"):
    rows = []
    for team, opponent, starter in ((away, home, "901"), (home, away, "902")):
        for i in range(9):
            rows.append({
                "batting_team": team, "pitching_team": opponent,
                "batter_id": str((100 if team == "A" else 200) + i),
                "pitcher_id": starter, "stand": "L" if i % 2 else "R",
                "onbase": float(i % 3 == 0), "tb": float(i % 3 == 0),
                "k": float(i % 3 == 1), "outs_made": 1.,
            })
    return {
        "pk": pk, "date": pd.Timestamp(date), "away": away, "home": home,
        "f5": np.array([2., 1.]), "late": np.array([1., 2.]), "full": np.array([3., 3.]),
        "lineups": {away: [str(100+i) for i in range(9)], home: [str(200+i) for i in range(9)]},
        "starters": {home: "901", away: "902"}, "pa": pd.DataFrame(rows),
    }


class OffenseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(PLAN_PATH.read_text())
        self.hands = {
            "901": {"status": "VERIFIED_PROFILE", "throws": "L"},
            "902": {"status": "VERIFIED_PROFILE", "throws": "R"},
        }
        self.games = [synthetic_game("one", "2024-04-01"),
                      synthetic_game("two", "2024-04-02"),
                      synthetic_game("three", "2024-04-02")]

    def test_same_date_labels_and_pa_outcomes_cannot_change_features(self):
        before = build_rows(self.games, self.plan, self.hands)
        changed = copy.deepcopy(self.games)
        for game in changed[1:]:
            game["f5"][:] = [19, 13]
            game["late"][:] = [7, 11]
            game["pa"][["onbase", "tb", "k"]] = [1., 4., 0.]
        after = build_rows(changed, self.plan, self.hands)
        fields = BASE_FEATURES + DELTA_FEATURES + PLATOON_FEATURES + ["offset"]
        pd.testing.assert_frame_equal(before[fields], after[fields])

    def test_future_rows_cannot_change_earlier_features(self):
        before = build_rows(self.games, self.plan, self.hands)
        future = synthetic_game("future", "2024-04-03")
        future["pa"][["onbase", "tb", "k"]] = [1., 4., 0.]
        after = build_rows(self.games + [future], self.plan, self.hands)
        fields = BASE_FEATURES + DELTA_FEATURES + PLATOON_FEATURES + ["offset"]
        pd.testing.assert_frame_equal(before[fields], after.iloc[:len(before)][fields])

    def test_timestamp_cutoff_includes_doubleheader_isolation(self):
        frame = build_rows(self.games, self.plan, self.hands)
        second_day = frame[frame.game_date.eq(pd.Timestamp("2024-04-02"))]
        self.assertTrue(second_day.max_feature_date.eq(pd.Timestamp("2024-04-01")).all())

    def test_unknown_hand_is_not_guessed(self):
        frame = build_rows(self.games, self.plan, {})
        self.assertTrue(frame.opposing_throws.eq("UNKNOWN").all())
        np.testing.assert_allclose(frame[[f"pitcher_specialist_{x}" for x in ("onbase", "tb", "k")]], 0.)

    def test_joint_nonnegative_increment_distribution(self):
        m5, ml = np.array([.2, 2.4, 8.]), np.array([.1, 1.8, 4.])
        f5, full = phase_distributions(m5, ml, self.plan)
        np.testing.assert_allclose(f5.sum(axis=1), 1, atol=5e-11)
        np.testing.assert_allclose(full.sum(axis=1), 1, atol=5e-11)
        np.testing.assert_allclose(f5 @ np.arange(f5.shape[1]), m5, atol=1e-7)
        np.testing.assert_allclose(full @ np.arange(full.shape[1]), m5 + ml, atol=1e-7)
        # Nonnegative late increments imply stochastic dominance, stronger than
        # only checking Full mean >= F5 mean.
        width = f5.shape[1]
        self.assertTrue((np.cumsum(full, axis=1)[:, :width] <= np.cumsum(f5, axis=1) + 1e-12).all())

    def test_wdl_symmetry_and_probability_mass(self):
        pmf, _ = phase_distributions(np.array([2.5, 2.5]), np.array([2., 2.]), self.plan)
        p = wdl_from_pmf(pmf[:1], pmf[1:])
        self.assertAlmostEqual(float(p.sum()), 1., places=9)
        self.assertAlmostEqual(float(p[0, 0]), float(p[0, 2]), places=12)
        self.assertGreater(p[0, 1], 0)

    def test_profile_parser_uses_throwing_field_and_static_identity(self):
        source = '<span id="x_playerProfile_lblName">선수</span><span id="x_playerProfile_lblPosition">투수(좌투우타)</span>'
        r = parse_profile(source)
        self.assertEqual(r["throws"], "L")
        self.assertEqual(r["name"], "선수")
        self.assertEqual(parse_profile(source.replace("좌투우타", "우언좌타"))["throws"], "R")
        self.assertEqual(parse_profile("release_pos_x=-2; stand=R")["throws"], "UNKNOWN")

    def test_archived_bot_half_encoding_recovers_both_starters(self):
        rows = []
        for inning in range(1, 7):
            for half in ("top", "bot"):
                for slot in range(3):
                    rows.append({
                        "game_pk": "archived", "game_date": pd.Timestamp("2024-04-01"),
                        "home_team": "B", "away_team": "A", "inning": inning,
                        "inning_topbot": half, "at_bat_number": len(rows)+1, "pitch_number": 1,
                        "batter": (100 if half == "top" else 200) + ((inning-1)*3+slot) % 9,
                        "pitcher": 901 if half == "top" else 902, "stand": "R",
                        "events": "field_out", "post_away_score": 0, "post_home_score": 0,
                        "outs_when_up": slot, "post_outs": slot+1,
                    })
        games, audit = prepare_games(pd.DataFrame(rows))
        self.assertEqual(audit["valid_score_games"], 1)
        self.assertEqual(games[0]["starters"], {"B": "901", "A": "902"})
        self.assertEqual(len(games[0]["lineups"]["A"]), 9)


if __name__ == "__main__":
    unittest.main()

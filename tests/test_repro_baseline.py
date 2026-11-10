"""Integrity tests that exercise leakage, mapping and probability semantics."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts/kbo_repro_baseline.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
CFG = json.loads((ROOT / "research/repro_baseline/protocol.json").read_text())


def game(pk, date, k_a=9., k_b=1.):
    stats_a = np.array([24., 5., 7., 1., k_a, 1.])
    stats_b = np.array([24., 8., 12., 4., k_b, 2.])
    bat = {"A": {str(i): np.array([3., 1., 1., 0., 0., 0.]) for i in range(9)},
           "B": {str(i+10): np.array([3., 1., 1., 0., 0., 0.]) for i in range(9)}}
    return {"game_pk": pk, "date": pd.Timestamp(date), "teams": {"away": "A", "home": "B"},
            "scores": {"away": [2, 4], "home": [3, 5]},
            "starter_ids": {"A": "SP_A", "B": "SP_B"},
            "starters": {"A": {"id": "SP_A", "stats": stats_a, "outs": 18},
                         "B": {"id": "SP_B", "stats": stats_b, "outs": 12}},
            "relief": {"A": np.array([12., 3., 4., 1., 3., 0.]),
                       "B": np.array([12., 3., 4., 1., 3., 0.])},
            "lineups": {"A": list(bat["A"]), "B": list(bat["B"])}, "hitters": bat}


class IntegrityTests(unittest.TestCase):
    def test_target_and_same_day_outcomes_cannot_change_features(self):
        gs = [game("1", "2026-05-01"), game("2", "2026-05-02"), game("3", "2026-05-02")]
        old = b.build_features(gs, CFG)
        mutated = copy.deepcopy(gs)
        mutated[1]["scores"]["away"] = [15, 25]
        mutated[1]["starters"]["A"]["stats"] *= 20
        mutated[1]["relief"]["B"] *= 40
        new = b.build_features(mutated, CFG)
        features = sorted(set(sum(b.model_columns(CFG).values(), [])) | {"offset"})
        pd.testing.assert_frame_equal(old[features], new[features])

    def test_future_data_cannot_change_prior_features(self):
        gs = [game("1", "2026-05-01"), game("2", "2026-05-02")]
        old = b.build_features(gs, CFG)
        new = b.build_features(gs + [game("3", "2026-06-01", 100, 0)], CFG)
        pd.testing.assert_frame_equal(old, new.iloc[:len(old)].reset_index(drop=True))

    def test_opposing_starter_mapping(self):
        f = b.build_features([game("1", "2026-05-01"), game("2", "2026-05-02")], CFG)
        d = f[(f.game_pk == "2") & (f.phase == "f5")].set_index("team")
        self.assertEqual(d.loc["A", "opposing_starter"], "SP_B")
        self.assertEqual(d.loc["B", "opposing_starter"], "SP_A")
        self.assertLess(d.loc["A", "opp_sp_kbb_delta"], d.loc["B", "opp_sp_kbb_delta"])
        self.assertLess(d.loc["A", "opp_sp_outs_delta"], d.loc["B", "opp_sp_outs_delta"])

    def test_full_is_nonnegative_sum_and_probability_mass_valid(self):
        f5 = b.nb_pmf(3.8, .54)
        late = b.nb_pmf(.2, .54)
        full = np.convolve(f5, late)
        mean5 = np.dot(np.arange(len(f5)), f5)
        meanfull = np.dot(np.arange(len(full)), full)
        self.assertGreater(meanfull, mean5)
        self.assertAlmostEqual(meanfull, 4., places=7)
        p = b.score_distribution(full, full)
        self.assertAlmostEqual(p["away"] + p["draw"] + p["home"], 1., places=9)
        self.assertAlmostEqual(sum(p["win1loss"]), 1., places=9)

    def test_negative_or_missing_means_raise(self):
        for mu in [-1., 0., float("nan")]:
            with self.assertRaises(ValueError):
                b.nb_pmf(mu, .54)

    def test_brier_definition_and_draw_semantics(self):
        ll, bs = b.probability_losses([.2, .3, .5], 2)
        self.assertAlmostEqual(bs, .38)
        ans = b.evaluate_game({"f5": np.array([2., 2.]), "late": np.array([1., 1.])},
                              {"f5": [1, 1], "full": [3, 3]}, CFG)
        self.assertNotIn("full_decided_logloss", ans)
        self.assertIn("f5_wdl_logloss", ans)

    def test_monthly_training_cannot_see_forecast_month_targets(self):
        gs = [game(str(i), f"2026-05-{i+1:02d}") for i in range(5)]
        gs += [game("june", "2026-06-01")]
        frame = b.build_features(gs, CFG)
        config = copy.deepcopy(CFG)
        config["minimum_training_games"] = 2
        config["baselines"] = {"baseline_team": CFG["baselines"]["baseline_team"]}
        config["additions_to_baseline_team_individually"] = {}
        old, _, _ = b.monthly_forecasts(frame, config)
        frame.loc[frame.date >= "2026-06-01", "target"] += 20
        new, _, _ = b.monthly_forecasts(frame, config)
        np.testing.assert_allclose(old.mu, new.mu, rtol=0, atol=0)
        self.assertTrue((pd.to_datetime(old.train_max_date) < old.date).all())

    def test_pairing_detects_model_game_mismatch(self):
        cfg = copy.deepcopy(CFG)
        rows = []
        for model, pk in [("baseline_team", "1"), ("addition", "2")]:
            for phase in ["f5", "late"]:
                for home in [0, 1]:
                    rows.append({"game_pk": pk, "model": model, "phase": phase, "is_home": home,
                                 "mu": 2., "target": 1, "date": pd.Timestamp("2026-06-01"),
                                 "train_max_date": "2026-05-30"})
        with self.assertRaisesRegex(ValueError, "different games"):
            b.game_predictions(pd.DataFrame(rows), cfg)


if __name__ == "__main__":
    unittest.main()

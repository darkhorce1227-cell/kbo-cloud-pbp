"""Safety and exact historical-CDF tests for the 2026 PS Expected-IP tail helper."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import forecast_ps_expected_ip_tail as tail


class TestPSExpectedIPTail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status, cls.contract = tail.load_canonical()
        cls.residuals, cls.validation = tail.load_verified_residuals()

    def forecast(self, **kwargs):
        return tail.build_tail_forecast(
            status=self.status, contract=self.contract,
            residuals=self.residuals, validation=self.validation,
            **kwargs,
        )

    def test_2026_training_source_audited(self):
        self.assertEqual(len(self.residuals), 284)
        self.assertAlmostEqual(sum(self.residuals)/len(self.residuals), -0.7598198773356355, places=12)
        self.assertEqual(self.validation["counts"]["eval_matched"], 256)
        self.assertEqual(self.validation["counts"]["eval_unavailable"], 2)

    def test_exact_curve_at_six_rs_ip(self):
        x=self.forecast(rs_ip=6.0, rs_starts=30, role="ESTABLISHED_STARTER")
        self.assertEqual(x["status"], "CHALLENGER_ONLY")
        p=x["event_probabilities"]
        self.assertAlmostEqual(p["IP_GE_5"], 0.6192982456140351, places=12)
        self.assertAlmostEqual(p["IP_GE_6"], 0.3385964912280702, places=12)
        self.assertAlmostEqual(p["IP_LT_4"], 0.22631578947368422, places=12)
        self.assertIsNone(x["operational_selected_probabilities"])

    def test_no_start_no_fallback(self):
        x=self.forecast(rs_ip=0, rs_starts=0, role="EMERGENCY_START")
        self.assertEqual(x["status"], "UNAVAILABLE")
        self.assertTrue(all(z is None for z in x["event_probabilities"].values()))
        self.assertIsNone(x["mean_challenger"]["challenger_mean_ip"])

    def test_cap_and_role_require_review(self):
        x=self.forecast(
            rs_ip=6.0, rs_starts=15, role="OPENER",
            pitch_cap=50, injury_news=True, short_rest=True, manager_leash="tight"
        )
        self.assertEqual(x["status"], "CONTEXT_REVIEW_REQUIRED")
        flags=x["mean_challenger"]["context_flags"]
        for f in ["ROLE_OPENER","PITCH_CAP_PRESENT","INJURY_OR_CURRENT_NEWS","SHORT_REST_OR_WORKLOAD","TIGHT_MANAGER_LEASH"]:
            self.assertIn(f, flags)
        self.assertIsNone(x["operational_selected_probabilities"])

    def test_unknown_role_requires_review(self):
        x=self.forecast(rs_ip=6.0, rs_starts=20)
        self.assertEqual(x["status"], "CONTEXT_REVIEW_REQUIRED")

    def test_champion_not_overwritten(self):
        x=self.forecast(rs_ip=6.0, rs_starts=20, role="ESTABLISHED_STARTER", champion_ip=6.5)
        self.assertEqual(x["mean_challenger"]["champion_ip"], 6.5)
        self.assertIsNone(x["mean_challenger"]["operational_selected_ip"])

    def test_probability_monotonicity_and_range(self):
        for ip in [3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 9.0]:
            p=self.forecast(rs_ip=ip, rs_starts=1, role="ESTABLISHED_STARTER")["event_probabilities"]
            self.assertGreaterEqual(p["IP_GE_5"], p["IP_GE_6"])
            self.assertTrue(all(0 < v < 1 for v in p.values()))


if __name__ == "__main__":
    unittest.main()

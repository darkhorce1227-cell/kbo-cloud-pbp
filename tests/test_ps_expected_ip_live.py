import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "forecast_ps_expected_ip_mean.py"
spec = importlib.util.spec_from_file_location("ps_eip_live", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestPostseasonExpectedIPLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status, cls.contract = mod.load_canonical()

    def test_canonical_shift_and_support(self):
        self.assertEqual(self.status["status"], "CLOSED_CANONICAL_STRICT_REPRODUCTION")
        self.assertTrue(math.isclose(self.status["canonical_2026_shift_ip"], -0.7598198773356355, abs_tol=1e-12))
        self.assertEqual(self.contract["historical_support"]["training_delta_eligible"], 284)
        self.assertEqual(self.contract["historical_support"]["walk_forward_evaluable"], 256)

    def test_established_starter_gets_challenger_only(self):
        out = mod.build_forecast(
            rs_ip=6.0,
            rs_starts=30,
            status=self.status,
            contract=self.contract,
            role="ESTABLISHED_STARTER",
            champion_ip=5.8,
            manager_leash="normal",
        )
        self.assertEqual(out["status"], "CHALLENGER_ONLY")
        self.assertTrue(math.isclose(out["challenger_mean_ip"], 5.2401801, abs_tol=1e-7))
        self.assertEqual(out["champion_ip"], 5.8)
        self.assertIsNone(out["operational_selected_ip"])

    def test_zero_same_season_starts_is_unavailable(self):
        out = mod.build_forecast(
            rs_ip=0.0,
            rs_starts=0,
            status=self.status,
            contract=self.contract,
            role="EMERGENCY_START",
        )
        self.assertEqual(out["status"], "UNAVAILABLE")
        self.assertEqual(out["reason"], "NO_SAME_SEASON_RS_START_HISTORY")
        self.assertIsNone(out["challenger_mean_ip"])

    def test_special_role_requires_context_review(self):
        out = mod.build_forecast(
            rs_ip=5.2,
            rs_starts=8,
            status=self.status,
            contract=self.contract,
            role="OPENER",
        )
        self.assertEqual(out["status"], "CONTEXT_REVIEW_REQUIRED")
        self.assertIn("ROLE_OPENER", out["context_flags"])
        self.assertIsNone(out["operational_selected_ip"])

    def test_current_context_outranks_history(self):
        out = mod.build_forecast(
            rs_ip=5.8,
            rs_starts=20,
            status=self.status,
            contract=self.contract,
            role="ESTABLISHED_STARTER",
            pitch_cap=70,
            injury_news=True,
            short_rest=True,
            manager_leash="tight",
        )
        self.assertEqual(out["status"], "CONTEXT_REVIEW_REQUIRED")
        self.assertIn("PITCH_CAP_PRESENT", out["context_flags"])
        self.assertIn("INJURY_OR_CURRENT_NEWS", out["context_flags"])
        self.assertIn("SHORT_REST_OR_WORKLOAD", out["context_flags"])
        self.assertIn("TIGHT_MANAGER_LEASH", out["context_flags"])


if __name__ == "__main__":
    unittest.main()

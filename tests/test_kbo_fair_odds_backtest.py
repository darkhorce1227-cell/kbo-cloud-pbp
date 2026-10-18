import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import backtest_kbo_fair_odds as mod

COLUMNS=[
    "game_pk","market_id","market_type","line","price_asof_kst",
    "game_start_kst","settlement_status","settled_outcome","outcome",
    "decimal_odds","model_probability","model_lock_kst","model_lock_ref",
]
START="2026-09-15T18:30:00+09:00"
PRE="2026-09-15T08:10:00+09:00"
LOCK="2026-09-15T07:55:00+09:00"


def make(gid="test-game",typ="FULL_ML2",status="SETTLED",win="HOME",model=True,stamp=PRE):
    odds={"HOME":1.80,"AWAY":2.20} if typ=="FULL_ML2" else {"HOME":2.00,"DRAW":3.60,"AWAY":3.50}
    probs={"HOME":0.6,"AWAY":0.4} if typ=="FULL_ML2" else {"HOME":0.5,"DRAW":0.2,"AWAY":0.3}
    return [{ "game_pk":gid,"market_id":gid+"-"+typ,
        "market_type":typ,"line":"","price_asof_kst":stamp,
        "game_start_kst":START,"settlement_status":status,
        "settled_outcome":win if status=="SETTLED" else "",
        "outcome":opt,"decimal_odds":str(odd),
        "model_probability":str(probs[opt]) if model else "",
        "model_lock_kst":LOCK if model else "",
        "model_lock_ref":"github:fake-fixture-lock" if model else ""}
        for opt,odd in odds.items()]


class TestMarketFairOdds(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/"odds.csv"

    def data(self, rows):
        with self.path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        return mod.load_events(self.path)

    def test_binary_no_vig_and_brier(self):
        ev=self.data(make())[0]
        self.assertAlmostEqual(sum(ev["fair_probabilities"].values()),1)
        self.assertAlmostEqual(ev["fair_probabilities"]["HOME"],(1/1.8)/(1/1.8+1/2.2))
        res=mod.analyze([ev])["markets"]["FULL_ML2"]
        self.assertEqual(res["settled"],1)
        self.assertEqual(res["model_comparison"]["n"],1)
        self.assertAlmostEqual(
            res["model_comparison"]["locked_model_brier"],
            (0.6-1)**2+(0.4-0)**2
        )
        self.assertIn("NOT_COMPUTED", res["model_comparison"]["ROI_status"])

    def test_three_way_draw_is_scored(self):
        ev=self.data(make(typ="F5_1X2",win="DRAW",model=False))[0]
        self.assertEqual(ev["settled_outcome"],"DRAW")
        self.assertEqual(len(ev["fair_probabilities"]),3)
        self.assertEqual(mod.analyze([ev])["markets"]["F5_1X2"]["settled"],1)

    def test_binary_draw_push_is_not_a_loss(self):
        xs=self.data(make(status="PUSH",win="",model=False))
        res=mod.analyze(xs)["markets"]["FULL_ML2"]
        self.assertEqual(res["settled"],0)
        self.assertEqual(res["push"],1)
        self.assertIsNone(res["market_multiclass_brier"])

    def test_future_price_rejected(self):
        with self.assertRaisesRegex(ValueError,"POST_START_PRICE_LEAKAGE"):
            self.data(make(stamp="2026-09-15T18:35:00+09:00"))

    def test_incomplete_outcome_book_rejected(self):
        with self.assertRaisesRegex(ValueError,"expected outcomes"):
            self.data(make()[:1])

    def test_missing_clean_pre_ref_rejected(self):
        rows=make()
        for r in rows:
            r["model_lock_ref"]=""
        with self.assertRaisesRegex(ValueError,"PRE LOCK provenance"):
            self.data(rows)

    def test_late_model_rejected(self):
        rows=make()
        for r in rows:
            r["model_lock_kst"]="2026-09-15T19:30:00+09:00"
        with self.assertRaisesRegex(ValueError,"POST_START_MODEL_LEAKAGE"):
            self.data(rows)

    def test_duplicate_price_versions_not_independent_games(self):
        a=make(model=False)
        b=make(model=False,stamp="2026-09-15T15:20:00+09:00")
        res=mod.analyze(self.data(a+b))["markets"]["FULL_ML2"]
        self.assertEqual(res["price_snapshots"],2)
        self.assertEqual(res["total_market_offerings"],1)

    def test_no_data_is_never_made_up(self):
        with self.assertRaisesRegex(ValueError,"No historical odds rows"):
            self.data([])


if __name__=="__main__":
    unittest.main()

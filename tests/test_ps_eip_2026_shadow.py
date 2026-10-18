"""Unit tests for strict 2026 first precommitted postseason Expected IP grading."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import score_2026_ps_eip_lock as mod


class Test2026PSShadow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d=Path(self.tmp.name)
        self.lock=d/"lock.json"
        self.audit=d/"audit.json"
        self.out=d/"score.json"
        self.payload=json.loads(mod.LOCK.read_text(encoding="utf-8"))
        self.lock.write_text(json.dumps(self.payload,ensure_ascii=False),encoding="utf-8")
        self.audit_payload={
            "validation":"PASS","target_date":"2026-10-18",
            "games":[{"game_pk":"TEST_PREGAME_GAME","away_team":"LG","home_team":"HT","status":"PASS"}]
        }
        self.audit.write_text(json.dumps(self.audit_payload),encoding="utf-8")
        self.relay={
            "awayLineup":{"pitcher":[{"seqno":1,"pcode":"61101","name":"임찬규","inn":"5 2/3","ballCount":92}]},
            "homeLineup":{"pitcher":[{"seqno":1,"pcode":"55633","name":"올러","inn":"6 1/3","ballCount":103}]}
        }
        self.commit=datetime.fromisoformat("2026-10-18T11:51:32+09:00")

    def run_score(self,commit=None,relay=None):
        with patch.object(mod,"_first_commit_datetime",return_value=(commit or self.commit)), \
             patch.object(mod.naver,"fetch_relay",return_value=(relay or self.relay)):
            return mod.score(self.lock,self.audit,self.out)

    def test_snapshot_historical_training_never_uses_2026(self):
        self.assertEqual(self.payload["training_postseasons"],"2016-2025")
        self.assertEqual(self.payload["ps_residual_n"],284)
        self.assertEqual(self.payload["rs_2026_completed_starter_appearances"],1440)
        self.assertEqual(len(self.payload["forecast_rows"]),2)
        a,b=self.payload["forecast_rows"]
        self.assertEqual((a["player_id"],a["rs_starts"],a["rs_total_outs"]),("61101",31,509))
        self.assertEqual((b["player_id"],b["rs_starts"],b["rs_total_outs"]),("55633",30,534))
        self.assertAlmostEqual(a["forecast"]["mean_challenger"]["challenger_mean_ip"],4.7132984,places=7)
        self.assertAlmostEqual(b["forecast"]["mean_challenger"]["challenger_mean_ip"],5.1735135,places=7)

    def test_first_real_pregame_lock_timestamp(self):
        self.assertLess(datetime.fromisoformat(self.payload["snapshot_created_at_kst"]),
                        datetime.fromisoformat(self.payload["first_pitch_kst"]))

    def test_score_fixture_valid_only_postgame(self):
        report=self.run_score()
        self.assertEqual(report["status"],"PASS")
        self.assertEqual(report["eligible_starter_rows"],2)
        self.assertEqual(report["rows"][0]["actual_outs"],17)
        self.assertEqual(report["rows"][1]["actual_outs"],19)
        self.assertEqual(report["rows"][0]["events"]["IP_GE_5"]["observed"],1)
        self.assertEqual(report["rows"][0]["events"]["IP_GE_6"]["observed"],0)
        self.assertEqual(report["promotion_decision"],"NO_MODEL_PROMOTION_FROM_ONE_GAME")

    def test_late_commit_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError,"PREGAME_LOCK_NOT_COMMITTED"):
            self.run_score(commit=datetime.fromisoformat("2026-10-18T14:00:01+09:00"))

    def test_changed_starter_cannot_be_scored(self):
        relay=json.loads(json.dumps(self.relay))
        relay["awayLineup"]["pitcher"][0]["pcode"]="DIFFERENT_ID"
        with self.assertRaisesRegex(ValueError,"STARTER_CHANGED"):
            self.run_score(relay=relay)

    def test_failed_official_audit_blocks_scoring(self):
        self.audit_payload["validation"]="FAIL"
        self.audit.write_text(json.dumps(self.audit_payload),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"OFFICIAL_GAME_AUDIT_NOT_PASS"):
            self.run_score()


if __name__=="__main__":
    unittest.main()

"""Safeguard 2026-10-26 KBO postseason PBP recovery; no fabricated data.

Run without live network. An upstream outage or vanished postseason schedule
must never publish a successful NO_GAMES or a falsely validated PBP dataset.
"""
import json
from datetime import date
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"cloud"))
import update_pbp as core


class TestRecovery(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name)
        data={"status":"BACKFILL_QUEUE_NOT_PBP_AND_NOT_OFFICIAL_SCORE_AUDIT",
              "targets":[{"target_date":"2026-10-26","data_ingested":False}]}
        (self.base/"pbp_recovery_targets.json").write_text(json.dumps(data),encoding="utf-8")

    def test_naver_date_and_season_timeouts_fail_closed(self):
        with patch.object(core,"STATE_DIR",self.base), \
             patch.object(core,"load_previous_status",return_value={"last_success_date":"2026-10-20"}), \
             patch.object(core,"fetch_live_daily_schedule",side_effect=TimeoutError("date read timeout")), \
             patch.object(core.schedule,"fetch",side_effect=TimeoutError("season read timeout")):
            r=core.run(date(2026,10,26),force=True)
        self.assertEqual(r.state,"UPSTREAM_SCHEDULE_UNAVAILABLE")
        self.assertEqual(r.validation,"FAIL")
        self.assertIn("date read timeout",r.reason)
        self.assertIn("season read timeout",r.reason)
        self.assertEqual(r.last_success_date,"2026-10-20")
        self.assertFalse(r.should_publish)

    def test_empty_upstream_schedule_does_not_hide_known_postseason_game(self):
        import pandas as pd
        empty=pd.DataFrame({"gameId":[],"gameDate":[]})
        with patch.object(core,"STATE_DIR",self.base), \
             patch.object(core,"load_previous_status",return_value={"last_success_date":"2026-10-20"}), \
             patch.object(core,"fetch_live_daily_schedule",return_value=empty):
            r=core.run(date(2026,10,26),force=True)
        self.assertEqual(r.state,"HISTORICAL_GAME_MISSING_FROM_UPSTREAM_SCHEDULE")
        self.assertEqual(r.validation,"FAIL")
        self.assertFalse(r.should_publish)

    def test_other_empty_day_is_legitimate_no_games(self):
        import pandas as pd
        empty=pd.DataFrame({"gameId":[],"gameDate":[]})
        with patch.object(core,"STATE_DIR",self.base), \
             patch.object(core,"load_previous_status",return_value={"last_success_date":"2026-10-20"}), \
             patch.object(core,"fetch_live_daily_schedule",return_value=empty):
            r=core.run(date(2026,10,24),force=True)
        self.assertEqual(r.state,"NO_GAMES")
        self.assertEqual(r.validation,"PASS")
        self.assertFalse(r.should_publish)

    def test_live_preflight_chooses_pending_10_26_before_newer_dates(self):
        with tempfile.TemporaryDirectory() as d:
            outfile=Path(d)/"output"
            env={**os.environ,
                 "GITHUB_OUTPUT":str(outfile),
                 "GITHUB_EVENT_NAME":"schedule",
                 "INPUT_TARGET_DATE":"",
                 "INPUT_FORCE":"false"}
            proc=subprocess.run([sys.executable,"cloud/preflight.py"],
                cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            text=outfile.read_text()
            self.assertIn("target_date=2026-10-26",text)
            self.assertIn("skip=false",text)


if __name__=="__main__":
    unittest.main()

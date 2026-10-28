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

    def test_postponed_20261026_real_game_and_four_pseudo_schedule_items(self):
        # Exact fields/IDs returned by the user's local successful 200 API test.
        import pandas as pd
        items=[
            {"gameId":"20261026SCBASEBALL1","gameDate":"2026-10-26","statusNum":4,"categoryName":"KBO"},
            {"gameId":"20261026KBO1","gameDate":"2026-10-26","statusNum":4,"categoryName":"KBO"},
            {"gameId":"20261026CHITCHIT1","gameDate":"2026-10-26","statusNum":4,"categoryName":"KBO"},
            {"gameId":"55551026HTSS02026","gameDate":"2026-10-26","statusNum":4,
             "categoryName":"KBO","statusInfo":"9회말"},
        ]
        class Response:
            def raise_for_status(self):pass
            def json(self):return {"result":{"games":items}}
        with patch.object(core.requests,"get",return_value=Response()):
            raw=core.fetch_live_daily_schedule(date(2026,10,26))
        self.assertEqual(raw["gameId"].tolist(),["55551026HTSS02026"])
        d=core.daily_schedule(raw,date(2026,10,26))
        self.assertEqual(core.get_expected_game_ids(d),["55551026HTSS02026"])
        self.assertEqual(core.get_final_game_ids(d),["55551026HTSS02026"])
        self.assertEqual(len(d),1)

    def test_roundcode_absent_allowed_explicit_noncompetitive_round_excluded(self):
        import pandas as pd
        rows=pd.DataFrame([
            {"gameId":"55551026HTSS02026","gameDate":"2026-10-26","roundCode":"",
             "statusNum":4,"statusInfo":"9회말"},
            {"gameId":"20261026HTSS02026","gameDate":"2026-10-26","roundCode":"kbo_e",
             "statusNum":4,"statusInfo":"9회말"},
        ])
        d=core.daily_schedule(rows,date(2026,10,26))
        self.assertEqual(d["gameId"].tolist(),["55551026HTSS02026"])

    def test_naver_game_ids_not_stale_oct25_date(self):
        self.assertTrue(core.is_real_kbo_game_id("55551026HTSS02026",2026))
        self.assertTrue(core.is_real_kbo_game_id("20261020HTLG02026",2026))
        self.assertFalse(core.is_real_kbo_game_id("20261026KBO1",2026))
        self.assertFalse(core.is_real_kbo_game_id("20261026SCBASEBALL1",2026))
        self.assertFalse(core.is_real_kbo_game_id("55551025HTSS02026",2025))

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

    def test_live_preflight_prioritizes_unresolved_recovery_only(self):
        # Previously hardcoded 10/26 forever, even after the 10/26 binary
        # audit legitimately PASSED. Check the *currently unresolved* tickets.
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
            output=outfile.read_text()
            self.assertIn("target_date=",output)
            record=json.loads((ROOT/"state"/"pbp_recovery_targets.json").read_text())
            unverified=[]
            for item in record.get("targets",[]):
                day=str(item["target_date"])
                if item.get("data_ingested") is not False:
                    continue
                audit_path=ROOT/"state"/f"postgame_score_audit_{day}.json"
                try:
                    audit=json.loads(audit_path.read_text())
                except (OSError,ValueError):
                    audit={}
                audited=(audit.get("validation")=="PASS"
                         and audit.get("target_date")==day
                         and int(audit.get("expected_games") or 0)>0
                         and audit.get("source")=="validated_parquet_binary")
                if not audited:
                    unverified.append(day)
            if unverified:
                self.assertIn("target_date="+min(unverified),output)
                self.assertIn("skip=false",output)
            if "2026-10-26" not in unverified:
                self.assertNotIn("target_date=2026-10-26\\n",output)


if __name__=="__main__":
    unittest.main()

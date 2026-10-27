"""Offline 10/26 Naver PBP importer archive safety gates (synthetic fixture).

Never create a fake game from this test. It exercises the verifier only:
matchup, date, SHA-256, nine innings, terminal GAME_END and score.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"cloud"))
from import_manual_pbp_20261026 import _extract,ID,DAY


def fixture(path:Path,change=None):
    run={
      "gameId":ID,"gameDate":DAY.isoformat(),"homeTeamCode":"SS",
      "awayTeamCode":"HT","homeTeamScore":1,"awayTeamScore":5,
      "statusCode":"RESULT","statusNum":4,"statusInfo":"9회말"}
    sched=json.dumps({"success":True,"result":{"games":[run]}}).encode()
    members={"schedule.json":sched}
    manifest={
      "schema":"kbo_naver_raw_relay_capture_v1","game_id":ID,
      "target_date_kst":DAY.isoformat(),"inning_count":9,
      "game_end_verified_in_relay":True,
      "schedule_sha256":hashlib.sha256(sched).hexdigest(),
      "innings":[]
    }
    for i in range(1,10):
        relay={
          "gameId":ID,
          "inningScore":{"away":{"1":"5"},"home":{"5":"1"}},
          "textRelays":[{"textOptions":[{"type":99}]}] if i==9
                       else [{"textOptions":[{"type":1}]}]
        }
        if change and change[0]=="wrong_game" and i==7:
            relay["gameId"]="BAD"
        b=json.dumps({"success":True,"result":{"textRelayData":relay}}).encode()
        name=f"relays/inning_{i:02d}.json"
        members[name]=b
        manifest["innings"].append({
            "inning":i,"file":name,"bytes":len(b),
            "sha256":hashlib.sha256(b).hexdigest(),
            "game_end_type_99":i==9
        })
    if change and change[0]=="bad_manifest_sha":
        manifest["innings"][3]["sha256"]="0"*64
    if change and change[0]=="missing_inning":
        del members["relays/inning_04.json"]
    members["manifest.json"]=json.dumps(manifest).encode()
    with ZipFile(path,"w") as z:
        for name,b in members.items():z.writestr(name.replace("/","\\"),b)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"test.zip"

    def test_all_nine_required_and_finished(self):
        fixture(self.path)
        game,schedule,manifest=_extract(self.path)
        self.assertEqual(sorted(game),list(range(1,10)))
        self.assertEqual(schedule["awayTeamScore"],5)
        self.assertEqual(manifest["inning_count"],9)

    def test_sha_tampering_rejected(self):
        fixture(self.path,("bad_manifest_sha",None))
        with self.assertRaisesRegex(RuntimeError,"Tampered"):
            _extract(self.path)

    def test_missing_inning_rejected(self):
        fixture(self.path,("missing_inning",None))
        with self.assertRaisesRegex(RuntimeError,"Unexpected archive members"):
            _extract(self.path)

    def test_wrong_relay_game_id_rejected(self):
        fixture(self.path,("wrong_game",None))
        with self.assertRaisesRegex(RuntimeError,"Wrong game relay"):
            _extract(self.path)


if __name__=="__main__":
    unittest.main()

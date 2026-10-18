from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import capture_kbo_proto_pregame_quotes as capture
import audit_kbo_pregame_quote_commits as audit

URL="https://sharppick.app/proto/2026/124/"
AT=datetime.fromisoformat("2026-10-18T06:55:00+00:00")
HTML='''<html><head><title>프로토 승부식 2026년도 124회차</title></head>
<table>
<tr><th colspan="7">10/18(일)</th></tr>
<tr data-sp="baseball"><td>14:00</td><td><a href="/match/old">LG 트윈스 대 KIA 타이거즈</a> KBO</td>
<td><span><span class="rd-lb">승</span> 2.10</span><span><span class="rd-lb">패</span> 1.55</span></td>
<td></td><td></td><td></td><td>4:3 승</td></tr>
<tr><th colspan="7">10/20(화)</th></tr>
<tr data-sp="baseball"><td>18:30</td><td><a href="/match/upcoming">LG 트윈스 대 KIA 타이거즈</a> ⚾ KBO</td>
<td><span><span class="rd-lb">승</span> 1.82<span class="dn">▼1.80</span></span>
<span><span class="rd-lb">패</span> 1.76<span class="up">▲1.78</span></span></td>
<td></td>
<td><div><b>-2.5</b><span><span class="rd-lb">승</span> 2.12</span>
<span><span class="rd-lb">패</span> 1.50</span></div></td>
<td><div><b>9.5</b><span><span class="rd-lb">언더</span> 1.85</span>
<span><span class="rd-lb">오버</span> 1.68</span></div></td><td>-</td></tr>
</table></html>'''


class QuoteSafety(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        rows,issues=capture.parse_quotes(HTML,URL,AT)
        self.assertEqual(issues,[])
        self.assertEqual(len(rows),3)
        self.rows=rows
        self.path=self.root/"state/pregame_market_quotes/2026-10-20_HT_LG/example.json"
        self.path.parent.mkdir(parents=True)
        row=dict(rows[0])
        row.update({
            "snapshot_fingerprint_sha256":capture._stable_fingerprint(row),
            "source_round_response_sha256":"source_sha",
            "original_prospective_label":"PUBLIC_PRICE_CAPTURE_REQUIRES_FIRST_COMMIT_TIME_GATE"})
        self.path.write_text(json.dumps(row,ensure_ascii=False),encoding="utf-8")

    def test_future_only_and_prices_ignore_old_quote_arrow(self):
        self.assertEqual({r["market_type"] for r in self.rows},
                         {"FULL_ML2","FULL_HC2","FULL_OU"})
        m=self.rows[0]
        self.assertEqual(m["decimal_odds"],{"HOME":1.82,"AWAY":1.76})
        self.assertEqual(m["market_id"],"2026-10-20|HT>LG|FULL_ML2|")
        self.assertEqual(m["scheduled_first_pitch_kst"],"2026-10-20T18:30:00+09:00")
        self.assertFalse(m["displayed_price_is_executable_at_betman"])
        self.assertFalse(m["is_clean_pre_model_prediction"])
        self.assertIsNone(m["future_game_pk"])

    def test_no_retroactive_capture_after_or_close_to_kickoff(self):
        for t in ["2026-10-20T09:11:00+00:00","2026-10-20T09:45:00+00:00"]:
            r,_=capture.parse_quotes(HTML,URL,datetime.fromisoformat(t))
            self.assertEqual(r,[])
        r,_=capture.parse_quotes(HTML,URL,datetime.fromisoformat("2026-10-20T09:09:00+00:00"))
        self.assertEqual(len(r),3)

    def test_unknown_or_partial_price_cannot_make_valid_market(self):
        bad=HTML.replace('1.82<span class="dn">','—<span class="dn">')
        r,errors=capture.parse_quotes(bad,URL,AT)
        self.assertEqual(r,[])
        self.assertEqual(len(errors),1)
        missing=HTML.replace('<span><span class="rd-lb">패</span> 1.76<span class="up">▲1.78</span></span>','')
        r,errors=capture.parse_quotes(missing,URL,AT)
        self.assertEqual(r,[])
        self.assertEqual(len(errors),1)

    def test_round_index_discovers_only_existing_links(self):
        s='<a href="/proto/2026/123/">123</a><a href="/proto/2026/122/">122</a>'
        self.assertEqual(capture.latest_round_links(s),
           ["https://sharppick.app/proto/2026/123/","https://sharppick.app/proto/2026/122/"])

    def test_pre_game_git_add_is_proof_of_public_quote_only(self):
        when=AT+timedelta(minutes=1)
        r=audit.audit_one(self.path,self.root,commit_time=when)
        self.assertTrue(r["public_quote_provenance_eligible"])
        self.assertFalse(r["confirmed_bettable_before_kickoff"])
        self.assertFalse(r["genuine_clean_pre_model"])

    def test_late_git_add_is_ineligible(self):
        when=datetime.fromisoformat("2026-10-20T09:30:00+00:00")
        r=audit.audit_one(self.path,self.root,commit_time=when)
        self.assertEqual(r["status"],"REJECT_LATE_GIT_COMMIT")
        self.assertFalse(r["public_quote_provenance_eligible"])

    def test_fake_bettable_evidence_rejected(self):
        row=json.loads(self.path.read_text())
        row["displayed_price_is_executable_at_betman"]=True
        self.path.write_text(json.dumps(row,ensure_ascii=False))
        with self.assertRaisesRegex(ValueError,"falsely"):
            audit.audit_one(self.path,self.root,commit_time=AT)

    def test_tampered_immutable_book_rejected(self):
        row=json.loads(self.path.read_text())
        row["decimal_odds"]["HOME"]=8.5
        self.path.write_text(json.dumps(row,ensure_ascii=False))
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            audit.audit_one(self.path,self.root,commit_time=AT)

    def test_quote_file_reuse_does_not_rewrite(self):
        class Robots:
            status_code=200
            content=b"User-agent: *\nAllow: /\n"
        class Client:
            headers={}
            def get(self,*args,**kwargs):return Robots()
        with patch.object(capture,"source_get",side_effect=[
                ('<a href="/proto/2026/124/">124</a>',AT,''),
                (HTML,AT,'Sun, 18 Oct 2026 06:55:00 GMT')]):
            x=capture.capture(out_dir=self.root/"capture",source_rounds=1,session=Client())
        self.assertEqual(len(x["new_quote_files"]),3)
        p=Path(x["new_quote_files"][0]["path"])
        before=p.read_bytes()
        with patch.object(capture,"source_get",side_effect=[
                ('<a href="/proto/2026/124/">124</a>',AT,''),
                (HTML,AT,'Sun, 18 Oct 2026 06:55:00 GMT')]):
            y=capture.capture(out_dir=self.root/"capture",source_rounds=1,session=Client())
        self.assertEqual(len(y["new_quote_files"]),0)
        self.assertEqual(y["already_identical_quote_files"],3)
        self.assertEqual(p.read_bytes(),before)


if __name__=="__main__":unittest.main()

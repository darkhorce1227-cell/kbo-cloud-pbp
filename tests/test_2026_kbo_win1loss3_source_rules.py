"""KBO official WIN1LOSS: ① means 1-run margin INCLUDING 0-run ties."""
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from collect_2026_betlive_win1loss3 import parse_round
from audit_2026_betlive_win1loss3_vs_pbp import actual_event, fair
from compare_2026_win1loss_market_vs_naive import prob_score


class TestKboWin1LossSemantics(unittest.TestCase):
    def test_official_categories_at_boundaries(self):
        for home,away,expected in [
            (4,4,"WITHIN_1"),(5,4,"WITHIN_1"),(4,5,"WITHIN_1"),
            (6,4,"HOME_2PLUS"),(4,6,"AWAY_2PLUS"),
            (9,2,"HOME_2PLUS"),(2,9,"AWAY_2PLUS")]:
            with self.subTest(home=home,away=away):
                self.assertEqual(actual_event(home,away),expected)

    def test_three_way_book_overround_and_class_probabilities(self):
        p,margin=fair({"HOME_2PLUS":2.50,"WITHIN_1":3.10,"AWAY_2PLUS":2.27})
        self.assertAlmostEqual(sum(p.values()),1,places=12)
        self.assertGreater(margin,0.10)
        self.assertLess(margin,0.25)
        b,l=prob_score(p,"WITHIN_1")
        self.assertGreater(b,0)
        self.assertGreater(l,0)

    def test_three_way_does_not_accept_two_way_book(self):
        with self.assertRaisesRegex(ValueError,"Invalid 3-way book"):
            fair({"HOME_2PLUS":1.70,"AWAY_2PLUS":2.00})

    def test_html_extraction_requires_exact_2026_game_pk(self):
        html="""<html><head><title>프로토 승부식 2026년 50회차</title></head>
        <body><table><tr><th>시각(KST)</th><th>리그</th><th>경기</th>
        <th>승</th><th>무</th><th>패</th></tr>
        <tr><td>04/29 18:30</td><td>KBO</td><td><strong>두산</strong>
        vs <strong>삼성</strong></td><td>2.50</td><td>3.10</td><td>2.27</td></tr>
        </table></body></html>"""
        ids={"20260429SSOB02026"}
        rows,issues,n=parse_round(html,50,ids,"2026-10-18T00:00:00Z")
        self.assertEqual(n,1)
        self.assertEqual(issues,[])
        self.assertEqual(len(rows),1)
        x=rows[0]
        self.assertEqual(x["market_type"],"FULL_WIN1LOSS_3WAY")
        self.assertEqual(x["game_pk"],"20260429SSOB02026")
        self.assertEqual(x["source_middle_header"],
                         "무_AS_PUBLISHED__INTERPRETED_WIN1LOSE_WITHIN_1")
        self.assertEqual(x["source_price_temporality"],
                         "HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED")
        self.assertEqual(x["inferred_outcome_from_pbp"],"")

    def test_html_cannot_force_nonexistent_team_date_game(self):
        html="""<html><head><title>프로토 승부식 2026년 50회차</title></head>
        <body><table><tr><th>승</th><th>무</th><th>패</th></tr>
        <tr><td>04/29 18:30</td><td>KBO</td><td>두산 vs 삼성</td>
        <td>2.50</td><td>3.10</td><td>2.27</td></tr></table></body></html>"""
        rows,issues,n=parse_round(html,50,set(),"2026-10-18T00:00:00Z")
        self.assertEqual(n,1)
        self.assertEqual(rows,[])
        self.assertEqual(len(issues),1)
        self.assertIn("Game PK not unique",issues[0]["reason"])


if __name__=="__main__":
    unittest.main()

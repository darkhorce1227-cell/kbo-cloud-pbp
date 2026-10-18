from datetime import date,timedelta
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import compare_2026_win1loss_market_vs_naive as mod


def toy_records():
    rows=[]
    for i in range(18):
        dt=(date(2026,4,20)+timedelta(days=i)).isoformat()
        for side in (0,1):
            rows.append({
                'game_date':dt,'game_pk':f'game_{i}_{side}',
                'market_type':'FULL_WIN1LOSS_3WAY',
                'outcome_source':'SHA_VERIFIED_2026_PBP',
                'source_price_temporality':'HISTORICAL_ARCHIVED_QUOTES_TIME_UNVERIFIED',
                'fair_home_2plus':.45,'fair_within_1':.25,'fair_away_2plus':.30,
                'verified_outcome':'HOME_2PLUS' if i%2==0 else 'AWAY_2PLUS'
            })
    return rows


class TestClimatologyComparison(unittest.TestCase):
    def test_uniform_brier_three_way(self):
        p={k:1/3 for k in mod.CLASS}
        b,l=mod.prob_score(p,'WITHIN_1')
        self.assertAlmostEqual(b,2/3)
        self.assertAlmostEqual(l,1.0986122886681098)

    def test_first_day_has_no_training_rows(self):
        x=mod.score(toy_records())
        self.assertEqual(x[0]['climatology_prior_data_n'],0)
        self.assertEqual(x[1]['climatology_prior_data_n'],0)
        self.assertEqual(x[2]['climatology_prior_data_n'],2)
        self.assertEqual(x[3]['climatology_prior_data_n'],2)
        self.assertEqual(x[2]['predicted_before_day_climatology'],
                         x[3]['predicted_before_day_climatology'])
        self.assertAlmostEqual(x[2]['predicted_before_day_climatology']['HOME_2PLUS'],11/32)
        self.assertEqual(len(x),36)

    def test_random_input_order_does_not_leak_outcomes(self):
        r=toy_records()
        normal=mod.score(r)
        reverse=mod.score(list(reversed(r)))
        a={v['game_pk']:v['predicted_before_day_climatology'] for v in normal}
        b={v['game_pk']:v['predicted_before_day_climatology'] for v in reverse}
        self.assertEqual(a,b)

    def test_provenance_disallowed(self):
        x=toy_records()
        x[0]['source_price_temporality']='PROSPECTIVE_CLEAN_PRE'
        with self.assertRaisesRegex(ValueError,'False or missing'):
            mod.score(x)


if __name__=='__main__':
    unittest.main()

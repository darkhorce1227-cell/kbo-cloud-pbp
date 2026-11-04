import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mu_phase_invariants import phase_issues,assert_phase_consistent

class CoherentTest(unittest.TestCase):
    def test_valid(self):
        m={'f5_away':1.7,'f5_home':3.8,'f5_total':5.5,
           'full_away':4.8,'full_home':5.9,'full_total':10.7}
        self.assertEqual(phase_issues(m),[])
        assert_phase_consistent(m)

    def test_known_v13_aug22_negative_late(self):
        m={'f5_away':1.7473283565841964,'f5_home':3.8726829232765687,
           'f5_total':1.7473283565841964+3.8726829232765687,
           'full_away':7.609333121244629,'full_home':3.100775292637182,
           'full_total':7.609333121244629+3.100775292637182}
        self.assertIn('IMPOSSIBLE_NEGATIVE_6PLUS_RUNS_HOME',phase_issues(m))
        with self.assertRaisesRegex(ValueError,'DISTRIBUTION_INCONSISTENT'):
            assert_phase_consistent(m)

    def test_total_mismatch(self):
        m={'f5_away':1.,'f5_home':2.,'f5_total':4.,
           'full_away':3.,'full_home':2.5,'full_total':5.5}
        self.assertIn('F5_TOTAL_MARGINAL_MISMATCH',phase_issues(m))

    def test_no_fields(self):
        self.assertIn('MISSING_FIELDS',phase_issues({'full_away':4.})[0])

if __name__=='__main__':
    unittest.main()

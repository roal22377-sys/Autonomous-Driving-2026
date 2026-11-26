import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from benchmark_localizer_truth_1126 import first_stable_time


class LocalizerRecoveryTimingTests(unittest.TestCase):
    def test_uses_elapsed_time_and_restarts_after_bad_pose(self):
        t = [0., .2, .5, .6, 1., 1.2]
        self.assertEqual(first_stable_time(t, [True, True, False, True, True, True], .1), .6)

    def test_short_right_censored_good_suffix_is_not_recovery(self):
        self.assertIsNone(first_stable_time([0., .2, .4], [True] * 3, 0.))

    def test_does_not_count_good_samples_before_fault_end(self):
        self.assertIsNone(first_stable_time([0., .2, .5, .6], [True] * 4, .5))

    def test_reports_start_after_confirming_half_second(self):
        self.assertEqual(first_stable_time([0., .2, .5], [True] * 3, 0.), 0.)


if __name__ == '__main__':
    unittest.main()

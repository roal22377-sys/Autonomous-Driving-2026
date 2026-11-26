import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
from analyze_sector_traces_1126 import sectors, sector_times, compare


class SectorTimingTests(unittest.TestCase):
    def test_wrapped_sector_constant_speed_and_partial_pass(self):
        time=np.arange(0.,31.,.1)
        progress=5.+2*time
        definitions=[dict(id=0,start_m=18.,length_m=4.,kind='tight_corner_sequence')]
        rows=sector_times(time,progress,20.,definitions)
        self.assertEqual(len(rows),3)
        for row in rows:self.assertAlmostEqual(row['seconds'],2.)
        self.assertTrue(all(r['projection_valid'] for r in rows))

    def test_backtracking_counts_elapsed_time_and_bad_projection_rejected(self):
        time=np.arange(7.)
        progress=np.array([0.,1.,2.,1.,2.,3.,4.])
        definition=[dict(id=0,start_m=1.,length_m=2.,kind='tight_corner_sequence')]
        rows=sector_times(time,progress,10.,definition,np.array([False,False,True,False,False,False]))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['seconds'],4.)
        self.assertEqual(rows[0]['backwards_m'],1.)
        self.assertFalse(rows[0]['projection_valid'])

    def test_sectors_partition_circular_path_without_double_counting(self):
        k=np.zeros(100);k[[0,1,40,41]]=1.
        defs=sectors(k,.5,before=1.,after=2.)
        self.assertEqual(len(defs),4)
        self.assertAlmostEqual(sum(r['length_m'] for r in defs),50.)
        occupancy=np.zeros(100,int)
        for r in defs:
            indices=(round(r['start_m']/.5)+np.arange(round(r['length_m']/.5)))%100
            occupancy[indices]+=1
        np.testing.assert_array_equal(occupancy,np.ones(100))

    def test_failed_episode_cannot_qualify_on_fast_prefix(self):
        row=dict(track_sha256='a',track='map',seed=1,cap=8.,start='center',definitions=[])
        self.assertFalse(compare(dict(row,success=True),dict(row,success=False))['qualified'])


if __name__=='__main__':unittest.main()

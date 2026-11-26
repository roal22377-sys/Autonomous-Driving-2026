import unittest
from types import SimpleNamespace
import numpy as np
from utils.driving_metrics import DrivingMetrics


class DrivingMetricsTests(unittest.TestCase):
    def test_phase_bins_do_not_depend_on_model_reward_configuration(self):
        line=SimpleNamespace(n=60,ds=.1,k=np.full(60,.07))
        def measure(reward_says_straight):
            env=SimpleNamespace(_line_ref=lambda:line,_reward_line_hint=20,max_steer=.5,timestep=.025,
                                vehicle_state=lambda:np.array([0.,0.,0.,3.,0.,0.,0.]),
                                driving_geometry=lambda:dict(curvature=.07,lateral=.03,heading=.01,
                                                             straight=reward_says_straight,future_curvature=.07))
            m=DrivingMetrics();m.update(env,np.array([.02,0.]));return m.result()
        before,after=measure(False),measure(True)
        self.assertEqual(before,after)
        self.assertEqual(set(before),{'straight'})
        self.assertEqual(before['straight']['protocol'],DrivingMetrics.PROTOCOL)


if __name__=='__main__':unittest.main()

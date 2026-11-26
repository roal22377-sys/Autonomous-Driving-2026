"""Nominal model/command plumbing checks, not controller performance claims."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(SRC), str(SRC / 'installation')]
from benchmark_observable_planner_1126 import pending_from_history
from nominal_vehicle_1126 import NominalVehicle


class ObservablePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train_il_s2r as IL
        cls.old_cwd = Path.cwd()
        os.chdir(SRC)
        cls.args = IL.get_args(['--task_config', 'configs/task/f1tenth_preview_1126.yaml'])
        cls.maps = IL.MAPS

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.old_cwd)

    def test_reset_padding_is_physical_zero(self):
        hist = np.zeros((12, 2))
        decode_speed = lambda c: .5 + (c + 1) * 3.75
        sq, vq = pending_from_history(hist, 0, 5, 1, lambda c: c*.5, decode_speed)
        self.assertEqual(sq, [0.]*5)
        self.assertEqual(vq, [0.])
        # An actual normalized-zero command must still decode to 4.25 m/s.
        _, vq = pending_from_history(hist, 1, 5, 1, lambda c: c*.5, decode_speed)
        self.assertEqual(vq, [4.25])

    def test_short_speed_queue_continues_accelerating_and_braking(self):
        plant = NominalVehicle(self.args)
        state = np.zeros(7); state[3] = 2.
        fast = plant.predict(state, [0.]*5, [4.], (0., 4.), .025)
        slow = plant.predict(state, [0.]*5, [0.], (0., 0.), .025)
        self.assertAlmostEqual(fast[3], 2. + 2.9*.125)
        self.assertAlmostEqual(slow[3], 2. - 3.7*.125)
        self.assertEqual(plant.predict(state, [], [], (0., 4.), .025), (0., 0., 0., 2.))

    def test_nominal_one_step_matches_environment(self):
        from utils.env import F1Wrapper
        args = self.args
        args.seed = 7; np.random.seed(7)
        path = 'installation/maps_real/Practice1126b.npz'
        if not Path(path).exists():
            self.skipTest('Generated Practice1126b test fixture is unavailable')
        env = F1Wrapper(args, self.maps, obs_noise=True, sim_mode='eval', tracks=[path])
        env.fixed_speed_cap = 8; env.start_mode = 'center'
        plant = NominalVehicle(args)
        braking = accelerating = count = 0
        try:
            env.reset(options={'track': path})
            for k in range(70):
                state = env.vehicle_state().copy()
                sq, vq = env.pending_commands()
                prediction = plant.step(state, sq[0], vq[0], args.timestep)
                braking += int(vq[0] < state[3] - .001)
                accelerating += int(vq[0] > state[3] + .001)
                _, _, term, trunc, _ = env.step([.1*np.sin(k*.3), 1. if k % 40 < 20 else -1.])
                difference = prediction - env.vehicle_state()
                difference[4] = (difference[4]+np.pi) % (2*np.pi) - np.pi
                np.testing.assert_allclose(difference, 0., atol=2e-6, rtol=0)
                count += 1
                if term or trunc:
                    break
        finally:
            env.close()
        self.assertGreaterEqual(count, 30)
        self.assertGreater(braking, 10)
        self.assertGreater(accelerating, 10)


if __name__ == '__main__':
    unittest.main()

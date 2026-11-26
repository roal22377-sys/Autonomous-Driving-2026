import unittest
from types import SimpleNamespace
import numpy as np
from utils.env import F1Wrapper


class StraightRewardTests(unittest.TestCase):
    def run_commands(self,reset,wave,speed=4.):
        e=F1Wrapper.__new__(F1Wrapper)
        e.STRAIGHT_STEER_COST=5.;e.STRAIGHT_V_REF=4.;e.STRAIGHT_ENTRY_RESET=reset
        e.timestep=.025;e.delta_s=speed*e.timestep;e.steer_ema=0.;e._straight_prev=False
        e.is_straight=False;e.on_straight=lambda:e.is_straight
        for _ in range(40):e.straight_weave_cost(1.)
        e.is_straight=True
        return np.array([e.straight_weave_cost(.1*np.sin(2*np.pi*4*i*e.timestep) if wave else 0.) for i in range(400)])

    def test_corner_memory_does_not_penalize_a_settled_straight(self):
        legacy=self.run_commands(False,False);fixed=self.run_commands(True,False)
        self.assertGreater(legacy[:10].sum(),1.)
        self.assertEqual(float(fixed.sum()),0.)

    def test_sustained_straight_weave_remains_penalized(self):
        legacy=self.run_commands(False,True)[-80:];fixed=self.run_commands(True,True)[-80:]
        self.assertGreater(float(fixed.mean()),.01)
        np.testing.assert_allclose(fixed,legacy,atol=1e-7)

    def test_slowing_does_not_reduce_penalty_per_meter(self):
        fast=self.run_commands(True,True,4.)[-80:].sum()/(80*.025*4.)
        slow=self.run_commands(True,True,2.)[-80:].sum()/(80*.025*2.)
        self.assertAlmostEqual(float(fast),float(slow),places=10)

    def test_lookbehind_checks_the_whole_interval(self):
        e=F1Wrapper.__new__(F1Wrapper);k=np.zeros(100);k[43]=.2
        line=SimpleNamespace(n=100,ds=.1,k=k,heading=np.zeros(100),xy=np.c_[np.arange(100)*.1,np.zeros(100)],
                             project=lambda x,y,hint=None:50)
        e._line_ref=lambda:line;e.vehicle_state=lambda:np.array([5.,0.,0.,3.,0.,0.,0.])
        e.elapsed=0.;e._reward_geom_cache=None;e._reward_line_hint=None
        e.reward_geometry=dict(straight_ahead=[0.,.5,1.,2.],straight_curvature=.08)
        e.STRAIGHT_BACK=0.;self.assertTrue(e.driving_geometry()['straight'])
        e.STRAIGHT_BACK=1.;e.elapsed+=.025
        self.assertFalse(e.driving_geometry()['straight'])
        self.assertEqual(e.driving_geometry()['future_curvature'],0.)


if __name__=='__main__':unittest.main()

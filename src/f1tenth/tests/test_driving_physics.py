import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'installation'))
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.integrator import RK4Integrator
from utils.driving_physics import path_normal_acceleration


class DrivingPhysicsTests(unittest.TestCase):
    def agent(self):
        return SimpleNamespace(model=DynamicModel.ST,steer_angle_vel=.7,accel=-1.5,
            params=dict(mu=.75,C_Sf=4.718,C_Sr=5.4562,lf=.19,lr=.15,h=.074,m=3.74,I=.04712,
                        s_min=-.5,s_max=.5,sv_min=-2.65,sv_max=2.65,v_switch=7.319,a_max=3.7,v_min=-8.,v_max=8.))

    def test_matches_acceleration_from_integrated_world_positions(self):
        agent=self.agent();integrator=RK4Integrator();h=1e-4;largest_proxy_error=0.
        for speed in (-1.,.2,1.,3.,5.):
            for steer in (-.4,0.,.4):
                for yaw_rate in (-2.,0.,2.):
                    state=np.array([1.,-.2,steer,speed,.4,yaw_rate,.12])
                    u=np.array([agent.steer_angle_vel,agent.accel])
                    plus=integrator.integrate(agent.model.f_dynamics,state,u,h,agent.params)
                    minus=integrator.integrate(agent.model.f_dynamics,state,u,-h,agent.params)
                    acceleration=(plus[:2]-2*state[:2]+minus[:2])/h**2
                    velocity=(plus[:2]-minus[:2])/(2*h)
                    normal=np.array([-velocity[1],velocity[0]])/np.linalg.norm(velocity)
                    measured=float(acceleration@normal)
                    expected=path_normal_acceleration(agent,state)
                    self.assertAlmostEqual(expected,measured,delta=.002)
                    largest_proxy_error=max(largest_proxy_error,abs(speed*yaw_rate-measured))
        # These transient maneuvers must actually distinguish the two signals.
        self.assertGreater(largest_proxy_error,3.)

    def test_rejects_a_state_without_slip_dynamics(self):
        with self.assertRaisesRegex(ValueError,'ST model'):
            path_normal_acceleration(self.agent(),np.zeros(5))


if __name__=='__main__':unittest.main()

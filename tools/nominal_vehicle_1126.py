"""Nominal model for offline observable-controller experiments, without env access."""
import numpy as np


class NominalVehicle:
    def __init__(self, args):
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.integrator import IntegratorType
        from f1tenth_gym.envs.action import CarAction
        keys = ('mu', 'C_Sf', 'C_Sr', 'h', 'm', 'I', 's_min', 's_max',
                'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max')
        self.params = {k: float(args[k]) for k in keys}
        self.params.update(lf=float(args.length_f), lr=float(args.length_r))
        self.a_acc = float(args.sim2real['a_max_nominal'])
        self.a_brake = float(args.sim2real['a_brake_nominal'])
        self.params['a_max'] = self.a_acc
        self.model = DynamicModel.from_string(args.model)
        self.integrator = IntegratorType.from_string(args.integrator)
        self.action = CarAction(list(args.control_input), params=self.params)

    def step(self, state, steer, speed, dt):
        # Select the nominal acceleration/braking limit for each predicted step.
        params = dict(self.params)
        params['a_max'] = self.a_brake if speed < state[3] - 1e-3 else self.a_acc
        accl, sv = self.action.act(action=(speed, steer), state=state, params=params)
        out = self.integrator.integrate(self.model.f_dynamics, np.asarray(state, dtype=float),
                                        np.array([sv, accl]), dt, params)
        out[4] %= 2*np.pi
        return out

    def predict(self, state, steer_queue, speed_queue, last_command, dt):
        out = np.array(state, dtype=float, copy=True)
        # When the short speed queue empties, keep applying its latest command.
        # The original kinematic fallback instead stopped updating speed.
        hold_steer = steer_queue[-1] if len(steer_queue) else last_command[0]
        hold_speed = speed_queue[-1] if len(speed_queue) else last_command[1]
        for k in range(max(len(steer_queue), len(speed_queue))):
            steer = steer_queue[k] if k < len(steer_queue) else hold_steer
            speed = speed_queue[k] if k < len(speed_queue) else hold_speed
            out = self.step(out, steer, speed, dt)
        return float(out[0]), float(out[1]), float(out[4]), float(out[3])


def nominal_planner_class(base):
    class NominalPlanner(base):
        def predict(self, state, steer_queue, speed_queue, dt):
            return self.nominal_vehicle.predict(state, steer_queue, speed_queue,
                                                self._last_cmd, dt)
    return NominalPlanner

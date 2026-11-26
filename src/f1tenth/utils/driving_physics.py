"""Acceleration diagnostics from the simulator's own state derivative."""
import numpy as np


DYNAMICS_PARAMETERS = ('mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I',
                       's_min', 's_max', 'sv_min', 'sv_max', 'v_switch',
                       'a_max', 'v_min', 'v_max')


def path_normal_acceleration(agent, state=None):
    """Acceleration normal to velocity, including changing slip in the ST model.

    Its velocity direction is yaw + slip, so V*yaw_rate alone omits the
    slip-rate term during a steering reversal. Use the same derivative and
    applied controls as the physics integrator; do not change the dynamics.
    """
    state = np.asarray(agent.state if state is None else state, dtype=np.float64)
    if state.shape != (7,):
        raise ValueError('path-normal acceleration requires the 7-state ST model')
    derivative = agent.model.f_dynamics(
        state, np.asarray([agent.steer_angle_vel, agent.accel], dtype=np.float64),
        *(agent.params[key] for key in DYNAMICS_PARAMETERS))
    # derivative[4] also handles the ST model's kinematic branch below 0.5 m/s,
    # where the stored yaw-rate state need not equal the active yaw derivative.
    return float(abs(state[3]) * (derivative[4] + derivative[6]))

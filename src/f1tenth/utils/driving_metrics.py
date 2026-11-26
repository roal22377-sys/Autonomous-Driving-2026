"""Phase metrics for comparing speed and stability alongside completion."""
import numpy as np


class DrivingMetrics:
    PROTOCOL = 'smoothed_raceline_008_v1'

    def __init__(self):
        self.sums = {}
        self.steer_ema = 0.0

    def update(self, env, action):
        g = env.driving_geometry()
        line = env._line_ref()
        j = env._reward_line_hint
        idx = (j - np.round(np.array([.5, 1., 2.]) / line.ds).astype(int)) % line.n
        past = float(np.max(np.abs(line.k[idx])))
        ahead = (j + np.round(np.array([0., .5, 1., 2.]) / line.ds).astype(int)) % line.n
        future = float(np.max(np.abs(line.k[ahead])))
        # Evaluation phases use one fixed geometry definition for all models;
        # changing a policy's reward configuration must not change its test bins.
        phase = ('hairpin' if abs(g['curvature']) > .45 else 'entry' if future > .45
                 else 'exit' if past > .45 else 'straight' if future < .08 else 'other')
        steer = float(action[0]) * env.max_steer
        self.steer_ema += (steer-self.steer_ema) * min(1., env.timestep/.5)
        v = float(env.vehicle_state()[3])
        row = self.sums.setdefault(phase, np.zeros(6))
        row += [1, v, g['lateral']**2, g['heading']**2, (steer-self.steer_ema)**2, env.timestep]

    def result(self):
        return {key:dict(protocol=self.PROTOCOL,samples=int(r[0]),speed=float(r[1]/r[0]),lateral_rms=float(np.sqrt(r[2]/r[0])),
                         heading_rms=float(np.sqrt(r[3]/r[0])),steer_weave_rms=float(np.sqrt(r[4]/r[0])),seconds=float(r[5]))
                for key,r in self.sums.items() if r[0] > 0}

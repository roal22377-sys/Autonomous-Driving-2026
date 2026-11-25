'''
    Track preview from a known map (plan B, src/f1tenth/MAP_TOOL.md): what the expert knows about the course ahead,
    from the car's pose in the map and the course's race line with a speed plan. Shared by the simulator
    (utils/env.py: true pose + simulated localization error) and the car (agent_node: localization pose), so both
    build the same features:

        [ e_y, e_psi, curvature at +d (d in curv_ahead), target speed at +d (d in speed_ahead), valid ]

        e_y      [m]   signed lateral offset from the race line (+ = left of it), clipped to +-2
        e_psi    [rad] heading error to the race line direction (+ = pointing left of it)
        curv     [1/m] race-line curvature d meters ahead (+ = left turn), clipped to +-3
        speed    [m/s] race-line speed plan d meters ahead (nominal car: grip share x mu, a_acc, a_brake),
                       capped by the speed limit the policy is given
        valid    1 = pose known, 0 = localization lost (all other features 0): the policy also learns to drive
                       on LiDAR alone

    Config: `track_preview` block of the task yaml (configs/task/f1tenth_preview.yaml).
    Relative quantities only (no map coordinates), so they mean the same on every course.
'''
import os

import numpy as np

DEFAULT = {
    'enabled': False,
    'curv_ahead': [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0],   # [m] along the race line
    'speed_ahead': [0.0, 5.0, 10.0, 20.0],                         # [m]
    # speed plan with the NOMINAL car (the car does not know the episode's grip either)
    'plan': {'grip': 0.65 * 9.81, 'grip_share': 0.7, 'a_acc': 2.9, 'a_brake': 3.7, 'v_min': 0.5, 'v_max': 8.0},
    # simulated localization error (sim only): slowly wandering offset + jitter, rare jumps, lost intervals
    'noise': {'xy_std': 0.05, 'yaw_std_deg': 1.0, 'corr_time': 0.5, 'jitter_xy': 0.01,
              'jump_prob': 0.0005, 'jump_std': 0.3, 'jump_time': 1.0,
              'lost_episode_prob': 0.15, 'lost_rate': 0.05, 'lost_len': [0.5, 3.0],
              'lost_full_prob': 0.0},     # episodes without any localization (LiDAR only from start to end)
}


def get_preview_config(args):
    c = dict(DEFAULT)
    user = dict(args.get('track_preview', None) or {})
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k] = {**c[k], **dict(v)}
        else:
            c[k] = v
    c['enabled'] = bool(c['enabled'])
    return c


def preview_dim(cfg):
    return (2 + len(cfg['curv_ahead']) + len(cfg['speed_ahead']) + 1) if cfg['enabled'] else 0


class TrackPreview:
    '''race line (closed, (N, 2) [m]) -> features for a pose. One per course.'''

    def __init__(self, raceline_xy, cfg, ds=0.1):
        # imported here: utils/observation.py imports this module for every model, and the car's LiDAR-only models
        # must not need scipy / numba (11/25 review)
        from scipy.spatial import cKDTree
        try:
            import numba  # noqa: F401
        except ImportError:
            # the car may not have numba: utils/raceline.py's @njit loop runs as plain Python then. A stub module
            # instead of an edit there: raceline.py's source is part of the track generator signature
            # (utils/track_gen.generator_signature), editing it marks every generated track as stale
            import sys
            import types
            stub = types.ModuleType('numba')
            stub.njit = lambda *a, **k: a[0] if (len(a) == 1 and callable(a[0]) and not k) else (lambda f: f)
            sys.modules['numba'] = stub
        from utils.raceline import resample_closed, curvature_smooth, speed_profile
        xy, L = resample_closed(np.asarray(raceline_xy, np.float64), ds)
        self.xy, self.L, self.ds = xy, float(L), float(L) / len(xy)
        self.n = len(xy)
        t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
        self.heading = np.arctan2(t[:, 1], t[:, 0])
        self.k = np.clip(curvature_smooth(xy, span=1.0), -3.0, 3.0)
        p = cfg['plan']
        self.v = speed_profile(xy, p['v_min'], p['v_max'], p['grip_share'] * p['grip'], p['a_acc'],
                               min(p['a_brake'], p['grip_share'] * p['grip']))
        self.tree = cKDTree(xy)
        self.cfg = cfg
        self.ck = np.round(np.asarray(cfg['curv_ahead']) / self.ds).astype(int)
        self.cv = np.round(np.asarray(cfg['speed_ahead']) / self.ds).astype(int)
        self.dim = preview_dim(cfg)

    WINDOW = 2.5        # [m] hint: nearest race-line point within this along the line from the last one

    def project(self, x, y, hint=None):
        '''nearest race-line point; with `hint` (the last index) only within WINDOW m along the line, so a
        neighbouring corridor's line behind a thin barrier (opposite direction) is not taken. None: global.'''
        if hint is not None:
            w = int(self.WINDOW / self.ds)
            idx = (int(hint) + np.arange(-w, w + 1)) % self.n
            d = np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y)
            k = int(np.argmin(d))
            if d[k] < 2.0:
                return int(idx[k])
        _, i = self.tree.query([x, y])
        return int(i)

    def lateral(self, x, y):
        '''signed offset from the race line [m] (+ = left of it), unclipped (simulator reward: utils/env.py LINE_COST)'''
        i = self.project(x, y)
        h = self.heading[i]
        return float(-np.sin(h) * (x - self.xy[i, 0]) + np.cos(h) * (y - self.xy[i, 1]))

    def features(self, x, y, yaw, speed_cap=None, valid=True, hint=None):
        if not valid:
            return np.zeros(self.dim, np.float32)
        i = self.project(x, y, hint)
        self.last_i = i
        h = self.heading[i]
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        e_y = float(np.clip(-np.sin(h) * dx + np.cos(h) * dy, -2.0, 2.0))
        e_psi = float((yaw - h + np.pi) % (2 * np.pi) - np.pi)
        kk = self.k[(i + self.ck) % self.n]
        vv = self.v[(i + self.cv) % self.n]
        if speed_cap is not None and speed_cap > 0:
            vv = np.minimum(vv, speed_cap)
        return np.concatenate([[e_y, e_psi], kk, vv, [1.0]]).astype(np.float32)


class LocalizationNoise:
    '''sim only: the pose the car's localization would report (per episode; step() once per control step)'''

    def __init__(self, cfg, rng, dt, train=True):
        n = cfg['noise']
        self.n, self.rng, self.dt, self.train = n, rng, dt, train
        self.a = np.exp(-dt / max(n['corr_time'], 1e-3))
        s = np.sqrt(1 - self.a ** 2)
        self.s_xy, self.s_yaw = n['xy_std'] * s, np.radians(n['yaw_std_deg']) * s
        self.off = rng.normal(0, [n['xy_std'], n['xy_std'], np.radians(n['yaw_std_deg'])])
        self.jump, self.jump_left = np.zeros(2), 0.0
        self.lost_left = 0.0
        self.lost_ep = train and rng.random() < n['lost_episode_prob']
        p_full = float(n.get('lost_full_prob', 0.0))
        # F1_LOC_LOST=1: evaluation of the LiDAR-only fallback (train_ppo_s2r.py --lost_eval)
        self.full_lost = os.environ.get('F1_LOC_LOST') == '1' or (train and p_full > 0 and rng.random() < p_full)

    def step(self, x, y, yaw):
        n, r = self.n, self.rng
        self.off = self.a * self.off + r.normal(0, [self.s_xy, self.s_xy, self.s_yaw])
        if self.train and self.jump_left <= 0 and r.random() < n['jump_prob']:
            self.jump, self.jump_left = r.normal(0, n['jump_std'], 2), n['jump_time']
        self.jump_left -= self.dt
        j = self.jump if self.jump_left > 0 else 0.0
        if self.lost_ep and self.lost_left <= 0 and r.random() < n['lost_rate'] * self.dt:
            self.lost_left = r.uniform(*n['lost_len'])
        self.lost_left -= self.dt
        jit = r.normal(0, n['jitter_xy'], 2)
        return (x + self.off[0] + jit[0] + (j[0] if np.ndim(j) else 0.0),
                y + self.off[1] + jit[1] + (j[1] if np.ndim(j) else 0.0),
                yaw + self.off[2], self.lost_left <= 0 and not self.full_lost)

'''
    Observation = processed lidar (utils/lidar.py) + optional vehicle speed.
    Shared by the simulator (utils/env.py) and the real car
    (f1tenth_stack/agent_node.py) so both build exactly the same vector.

        [ lidar bin 0, ..., lidar bin N-1, (speed [m/s]), (action history), (speed limit [m/s]) ]

        action history (action_history.length = H > 0): the last H commands the policy sent,
        newest first, each [steer, speed] normalized to [-1, 1] -> 2*H values. The steering
        reacts ~0.2 s late on the real car, so the policy needs to know what it already
        commanded to avoid over-steering (weaving).

        speed  sim : obs_dict['linear_vels_x'][0]
               real: /odom twist.twist.linear.x (vesc_to_odom_node)

    Settings come from configs/task/f1tenth.yaml (`lidar`, `speed_obs`).
    Configs without `speed_obs` (older models) get lidar only.
'''

import numpy as np

from utils.lidar import get_lidar_config, process_scan, add_scan_noise

DEFAULT_SPEED_CONFIG = {
    'enabled': False,
    'clip': [0.0, 5.0],         # [m/s]
    'noise_std': 0.0,           # [m/s], sim training data only
    'scale_range': [1.0, 1.0],  # per-episode reading scale error, sim training data only
}


def get_speed_config(args):
    cfg = dict(DEFAULT_SPEED_CONFIG)
    cfg.update(dict(args.get('speed_obs', None) or {}))
    cfg['enabled'] = bool(cfg['enabled'])
    cfg['clip'] = [float(x) for x in cfg['clip']]
    cfg['noise_std'] = float(cfg['noise_std'])
    cfg['scale_range'] = [float(x) for x in cfg['scale_range']]
    return cfg


def get_history_config(args):
    return {'length': int((args.get('action_history', None) or {}).get('length', 0))}


def get_speed_limit_config(args):
    c = dict(args.get('speed_limit', None) or {})
    return {'enabled': bool(c.get('enabled', False)), 'max': float(args.get('max_speed', 8.0))}


def get_obs_config(args):
    '''Lidar + speed + action history + speed limit settings; checks that args.obs_dim matches them.'''
    cfg = {'lidar': get_lidar_config(args), 'speed': get_speed_config(args), 'history': get_history_config(args),
           'speed_limit': get_speed_limit_config(args)}
    obs_dim = cfg['lidar']['num_beams'] + (1 if cfg['speed']['enabled'] else 0) + 2*cfg['history']['length'] \
              + (1 if cfg['speed_limit']['enabled'] else 0)
    if args.obs_dim != obs_dim:
        raise ValueError(f"obs_dim is {args.obs_dim} but lidar.num_beams ({cfg['lidar']['num_beams']})"
                         f" + speed_obs ({int(cfg['speed']['enabled'])}) + 2*action_history ({cfg['history']['length']})"
                         f" + speed_limit ({int(cfg['speed_limit']['enabled'])})"
                         f" = {obs_dim}. Fix configs/task/f1tenth.yaml.")
    return cfg


class ActionHistory:
    '''Last H normalized commands [steer, speed], newest first. Same class on sim and car.'''
    def __init__(self, length):
        self.length = int(length)
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.length, 2), dtype=np.float32)

    def push(self, action):
        if self.length > 0:
            self.buf = np.roll(self.buf, 1, axis=0)
            self.buf[0] = np.clip(np.asarray(action, dtype=np.float32)[:2], -1.0, 1.0)

    def vector(self):
        return self.buf.reshape(-1)


def make_observation(raw_scan, speed, cfg, history=None, speed_limit=None):
    '''
        raw_scan:    >= 1080 lidar ranges [m]
        speed:       longitudinal speed [m/s] (ignored if speed_obs is disabled)
        history:     ActionHistory (required if action_history.length > 0)
        speed_limit: cap on the commanded speed [m/s]; None = no cap (max_speed)
    '''
    obs = process_scan(raw_scan, **cfg['lidar'])
    if cfg['speed']['enabled']:
        speed = float(speed)
        if not np.isfinite(speed):
            speed = 0.0
        speed = np.clip(speed, *cfg['speed']['clip'])
        obs = np.append(obs, speed)
    if cfg.get('history', {}).get('length', 0) > 0:
        obs = np.concatenate([obs, history.vector()])
    if cfg.get('speed_limit', {}).get('enabled', False):
        lim = cfg['speed_limit']['max'] if speed_limit is None or speed_limit <= 0 else min(float(speed_limit), cfg['speed_limit']['max'])
        obs = np.append(obs, lim)
    return obs.astype(np.float32)


def add_obs_noise(raw_scan, speed, rng, cfg, speed_scale=1.0, lidar_override=None):
    '''
        Sim-only augmentation for training data, applied before make_observation().
        speed_scale: per-episode multiplicative error, sampled from speed.scale_range at reset.
        lidar_override: per-episode noise parameters (utils/env.py randomizes them).
    '''
    lidar = dict(cfg['lidar'])
    lidar.update(lidar_override or {})
    raw_scan = add_scan_noise(raw_scan, rng, **lidar)
    speed = speed * speed_scale + rng.normal(0.0, cfg['speed']['noise_std'])
    return raw_scan, speed

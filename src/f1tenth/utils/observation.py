'''
    Observation = processed lidar (utils/lidar.py) + optional vehicle speed.
    Shared by the simulator (utils/env.py) and the real car
    (f1tenth_stack/agent_node.py) so both build exactly the same vector.

        [ lidar bin 0, ..., lidar bin N-1, (speed [m/s]) ]

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


def get_obs_config(args):
    '''Lidar + speed settings; checks that args.obs_dim matches them.'''
    cfg = {'lidar': get_lidar_config(args), 'speed': get_speed_config(args)}
    obs_dim = cfg['lidar']['num_beams'] + (1 if cfg['speed']['enabled'] else 0)
    if args.obs_dim != obs_dim:
        raise ValueError(f"obs_dim is {args.obs_dim} but lidar.num_beams ({cfg['lidar']['num_beams']})"
                         f" + speed_obs ({int(cfg['speed']['enabled'])}) = {obs_dim}. Fix configs/task/f1tenth.yaml.")
    return cfg


def make_observation(raw_scan, speed, cfg):
    '''
        raw_scan: >= 1080 lidar ranges [m]
        speed:    longitudinal speed [m/s] (ignored if speed_obs is disabled)
    '''
    obs = process_scan(raw_scan, **cfg['lidar'])
    if cfg['speed']['enabled']:
        speed = float(speed)
        if not np.isfinite(speed):
            speed = 0.0
        speed = np.clip(speed, *cfg['speed']['clip'])
        obs = np.append(obs, speed)
    return obs.astype(np.float32)


def add_obs_noise(raw_scan, speed, rng, cfg, speed_scale=1.0):
    '''
        Sim-only augmentation for training data, applied before make_observation().
        speed_scale: per-episode multiplicative error, sampled from speed.scale_range at reset.
    '''
    raw_scan = add_scan_noise(raw_scan, rng, **cfg['lidar'])
    speed = speed * speed_scale + rng.normal(0.0, cfg['speed']['noise_std'])
    return raw_scan, speed

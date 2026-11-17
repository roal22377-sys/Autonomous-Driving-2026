'''
    LiDAR preprocessing shared by the simulator (utils/env.py) and the real car
    (f1tenth_stack/agent_node.py). Both sides call process_scan() with the same
    config so the policy sees identical observations in sim and real.

    raw scan (1080) -> invalid values (NaN/inf/too close) set to max_range
                    -> clip to [0, max_range]
                    -> pool every (1080 / num_beams) beams into one value

    Settings come from the `lidar` block of configs/task/f1tenth.yaml
    (copied to results/.../backup/f1tenth.yaml, which agent_node.py reads).
    If the block is missing (older models), defaults keep the raw 1080 scan.
'''

import numpy as np

RAW_BEAMS = 1080    # sim beams; the real Hokuyo gives 1081, agent_node uses the first 1080

DEFAULT_LIDAR_CONFIG = {
    'num_beams': RAW_BEAMS,     # no downsampling
    'max_range': 30.0,          # sim ScanSimulator2D default
    'min_valid': 0.0,
    'pooling': 'min',
    'noise_std': 0.0,
    'dropout_prob': 0.0,
    'spike_short_prob': 0.0,    # isolated return much shorter than the wall (dust, edge), sim only
    'spike_long_prob': 0.0,     # isolated return much longer (mixed pixel / missed edge), sim only
    'rotation_deg': 0.0,        # per-episode mount yaw error, sim only
}

_POOLING = {
    'min': np.min,
    'mean': np.mean,
    'median': np.median,
}


def get_lidar_config(args):
    '''Lidar settings from task args, filling defaults for missing keys.'''
    cfg = dict(DEFAULT_LIDAR_CONFIG)
    cfg.update(dict(args.get('lidar', None) or {}))
    cfg['num_beams'] = int(cfg['num_beams'])
    cfg['pooling'] = str(cfg['pooling'])
    for key in ('max_range', 'min_valid', 'noise_std', 'dropout_prob', 'spike_short_prob', 'spike_long_prob', 'rotation_deg'):
        cfg[key] = float(cfg[key])

    if RAW_BEAMS % cfg['num_beams'] != 0:
        raise ValueError(f"lidar.num_beams ({cfg['num_beams']}) must divide {RAW_BEAMS}.")
    if cfg['pooling'] not in _POOLING:
        raise ValueError(f"lidar.pooling must be one of {list(_POOLING)}, got '{cfg['pooling']}'.")
    return cfg


def process_scan(raw_scan, num_beams, max_range, min_valid, pooling, **_):
    '''
        raw_scan: array-like with at least 1080 ranges [m]
        returns:  (num_beams,) float32 array in [0, max_range]
    '''
    scan = np.asarray(raw_scan, dtype=np.float32).reshape(-1)
    if scan.shape[0] < RAW_BEAMS:
        raise ValueError(f'expected at least {RAW_BEAMS} lidar ranges, got {scan.shape[0]}.')
    scan = scan[:RAW_BEAMS]

    # invalid returns (NaN/inf, or closer than the sensor can measure) -> "nothing there"
    scan = np.where(np.isfinite(scan), scan, max_range)
    scan[scan < min_valid] = max_range
    scan = np.clip(scan, 0.0, max_range)

    return _POOLING[pooling](scan.reshape(num_beams, -1), axis=1).astype(np.float32)


def add_scan_noise(raw_scan, rng, noise_std, dropout_prob, spike_short_prob=0.0, spike_long_prob=0.0,
                   rotation_beams=0, **_):
    '''
        Sim-only augmentation on the raw scan, before process_scan().
        Matches the real Hokuyo measured on the car (logs/run_1117_14*, stationary frames):
          gaussian range noise ~0.005-0.008 m, invalid returns 0-0.25 %,
          isolated short spikes ~0.02-0.06 %, isolated long spikes ~0.02-0.09 %.
        rotation_beams: per-episode mount yaw error, as a shift in beams (0.25 deg / beam).
    '''
    scan = np.asarray(raw_scan, dtype=np.float32).reshape(-1).copy()
    if rotation_beams:
        scan = np.roll(scan, int(rotation_beams))
    if noise_std > 0:
        scan += rng.normal(0.0, noise_std, size=scan.shape).astype(np.float32)
    if spike_short_prob > 0:
        m = rng.random(scan.shape) < spike_short_prob
        scan[m] *= rng.uniform(0.1, 0.7, size=int(m.sum())).astype(np.float32)
    if spike_long_prob > 0:
        m = rng.random(scan.shape) < spike_long_prob
        scan[m] += rng.uniform(0.5, 5.0, size=int(m.sum())).astype(np.float32)
    if dropout_prob > 0:
        scan[rng.random(scan.shape) < dropout_prob] = np.inf
    return scan

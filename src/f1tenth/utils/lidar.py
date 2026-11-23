'''
    LiDAR preprocessing shared by the simulator (utils/env.py) and the real car
    (f1tenth_stack/agent_node.py). Both sides call process_scan() with the same
    config so the policy sees identical observations in sim and real.

    raw scan (1080) -> invalid values (NaN/inf/too close) set to max_range
                    -> clip to [0, max_range]
                    -> pool every (1080 / num_beams) beams into one value
                       or, with `zones`, bins of different widths (dense ahead, coarse behind)

    zones (optional, symmetric about straight ahead), from the front outwards:
        zones: [[30, 2.5], [90, 5.0], [135, 7.5]]
        = |angle| 0-30 deg in 2.5 deg bins, 30-90 in 5 deg bins, 90-135 in 7.5 deg bins
          -> 24 + 2*12 + 2*6 = 60 bins. num_beams must equal the resulting count.
        Raw beams are 0.25 deg apart (1080 over 270 deg, beam 0 = -135 deg = right), so every zone
        edge and bin width must be a multiple of 0.25 deg; the last zone must end at 135.

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
    'zones': None,              # non-uniform bins, see the top of this file
}
FOV_DEG = 270.0
BEAM_DEG = FOV_DEG / RAW_BEAMS  # 0.25


def zone_edges(zones):
    '''
        zones [[end_deg, bin_deg], ...] (from the front outwards, symmetric) ->
        bin edges as raw beam indices (n_bins + 1,), from the right end (-135 deg) to the left end.
    '''
    half = FOV_DEG / 2
    left, start = [], 0.0               # bin edges in degrees, 0 .. 135 (left half)
    for end, width in zones:
        end, width = float(end), float(width)
        n = (end - start) / width
        if end <= start or abs(n - round(n)) > 1e-6:
            raise ValueError(f'lidar.zones: zone {start:g}-{end:g} deg is not a whole number of {width:g} deg bins.')
        if abs(width / BEAM_DEG - round(width / BEAM_DEG)) > 1e-6:
            raise ValueError(f'lidar.zones: bin width {width:g} deg is not a multiple of {BEAM_DEG} deg.')
        left += list(start + width * np.arange(1, int(round(n)) + 1))
        start = end
    if abs(start - half) > 1e-6:
        raise ValueError(f'lidar.zones: the last zone must end at {half:g} deg, got {start:g}.')
    deg = np.concatenate([-np.array(left[::-1]), [0.0], np.array(left)])     # -135 .. 135
    return np.round((deg + half) / BEAM_DEG).astype(int)


def bin_centers_deg(cfg):
    '''angle [deg] of the middle of every bin (for plots / analysis)'''
    edges = cfg['edges'] if cfg.get('edges') is not None else np.linspace(0, RAW_BEAMS, cfg['num_beams'] + 1)
    e = np.asarray(edges, float)
    return (e[:-1] + e[1:]) / 2 * BEAM_DEG - FOV_DEG / 2

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

    cfg['edges'] = None
    if cfg.get('zones'):
        cfg['zones'] = [[float(a), float(b)] for a, b in cfg['zones']]
        cfg['edges'] = zone_edges(cfg['zones'])
        n = len(cfg['edges']) - 1
        if cfg['num_beams'] != n:
            raise ValueError(f"lidar.zones give {n} bins but lidar.num_beams is {cfg['num_beams']}.")
    elif RAW_BEAMS % cfg['num_beams'] != 0:
        raise ValueError(f"lidar.num_beams ({cfg['num_beams']}) must divide {RAW_BEAMS}.")
    if cfg['pooling'] not in _POOLING:
        raise ValueError(f"lidar.pooling must be one of {list(_POOLING)}, got '{cfg['pooling']}'.")
    return cfg


def process_scan(raw_scan, num_beams, max_range, min_valid, pooling, edges=None, **_):
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

    if edges is None:
        return _POOLING[pooling](scan.reshape(num_beams, -1), axis=1).astype(np.float32)
    # non-uniform bins
    if pooling == 'min':
        return np.minimum.reduceat(scan, edges[:-1]).astype(np.float32)
    if pooling == 'mean':
        return (np.add.reduceat(scan, edges[:-1]) / np.diff(edges)).astype(np.float32)
    return np.array([np.median(scan[a:b]) for a, b in zip(edges[:-1], edges[1:])], dtype=np.float32)


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
        # shift without wrapping (the scan covers 270 deg, its ends are not neighbours)
        k = int(rotation_beams)
        scan = np.concatenate([np.full(k, scan[0]), scan[:-k]]) if k > 0 else np.concatenate([scan[-k:], np.full(-k, scan[-1])])
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

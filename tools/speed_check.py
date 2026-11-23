#!/usr/bin/env python3
'''
Is /odom speed (VESC erpm / speed_to_erpm_gain) the car's true speed?

The true speed comes from lidar scan matching (tools/steer_delay.py icp_pose): displacement between
scans STRIDE apart / time. Only nearly straight driving is used (the lidar sits ahead of the rear axle,
so it moves faster than the wheels in a turn), and only scans with walls in two directions
(in a featureless corridor the along-corridor shift is not observable).

usage (after tools/bag_extract.py):
    python3 tools/speed_check.py logs/run_1123_1350_extracted logs/run_1123_1402_extracted
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer_delay import scan_points, icp_pose  # noqa: E402

STRIDE = 16             # scans between matched pairs (0.4 s at 40 Hz; shorter is too noisy)
MAX_YAW_RATE = 0.4      # [rad/s] nearly straight only
MIN_MATCH = 0.7         # matched point fraction
MIN_CONDITION = 0.08    # smallest / largest eigenvalue of the wall-normal matrix (corridor -> ~0)
VESC_GAIN = 4180.0      # speed_to_erpm_gain in vesc.yaml (for the suggested value)


def normal_condition(P):
    '''how well a 2D scan pins down translation in both directions: eigenvalue ratio of sum(n n^T)'''
    if len(P) < 20:
        return 0.0
    d = P[2:] - P[:-2]                      # tangent from neighbouring points (scan order)
    gap = np.linalg.norm(d, axis=1)
    d = d[(gap > 1e-3) & (gap < 0.3)]
    if len(d) < 20:
        return 0.0
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]]
    w = np.linalg.eigvalsh(n.T @ n)
    return float(w[0] / w[1])


def analyse(log_dir, modes):
    d = pd.read_csv(os.path.join(log_dir, 'driving.csv'))
    z = np.load(os.path.join(log_dir, 'scan.npz'))
    R, ts = z['ranges'], z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
    mode = d['mode'].values[np.clip(np.searchsorted(d.t.values, ts), 0, len(d) - 1)]
    v_odom = np.interp(ts, d.t, d.odom_speed)
    rows = []
    for i in range(0, len(ts) - STRIDE, 3):
        j = i + STRIDE
        if mode[i] not in modes or mode[j] not in modes or min(v_odom[i], v_odom[j]) < 0.8:
            continue
        # pushed against a wall: the wheels spin but the car doesn't move
        if np.nanmin(np.where(R[i:j + 1] > 0.05, R[i:j + 1], np.inf)) < 0.25:
            continue
        A, B = scan_points(R[i], ang), scan_points(R[j], ang)
        th, tr, frac = icp_pose(A, B)
        dt = ts[j] - ts[i]
        if not np.isfinite(th) or frac < MIN_MATCH or abs(th / dt) > MAX_YAW_RATE:
            continue
        cond = normal_condition(A)
        if cond < MIN_CONDITION:
            continue
        rows.append(dict(t=ts[i], v_lidar=np.linalg.norm(tr) / dt,
                         v_odom=float(np.mean(v_odom[i:j + 1])), cond=cond))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dirs', nargs='+')
    p.add_argument('--modes', nargs='+', default=['auto', 'manual'])
    a = p.parse_args()
    allr = []
    for ld in a.log_dirs:
        df = analyse(ld, set(a.modes))
        name = os.path.basename(os.path.normpath(ld)).replace('_extracted', '')
        if df.empty:
            print(f'{name}: no usable straight samples')
            continue
        df['ratio'] = df.v_lidar / df.v_odom
        allr.append(df)
        print(f'{name}: {len(df)} samples, lidar/odom median {df.ratio.median():.3f} '
              f'(p25 {df.ratio.quantile(.25):.3f}, p75 {df.ratio.quantile(.75):.3f})')
    if not allr:
        return
    df = pd.concat(allr)
    print('\nby odom speed:')
    g = df.groupby(pd.cut(df.v_odom, [0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 9.0]), observed=True)
    print(g.ratio.agg(['median', 'count']).round(3).to_string())
    r = df.ratio.median()
    print(f'\nall: true speed = {r:.3f} x odom  ->  speed_to_erpm_gain {VESC_GAIN:.0f} would be '
          f'{VESC_GAIN / r:.0f} for odom to read the true speed')


if __name__ == '__main__':
    main()

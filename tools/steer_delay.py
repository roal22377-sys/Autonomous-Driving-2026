#!/usr/bin/env python3
'''
Steering delay (command -> actual yaw response) from a driving log, no policy involved.

The car's real yaw rate comes from lidar scan matching (ICP between consecutive scans),
so it does not depend on odom (whose angular velocity is computed from the servo command).
The commanded yaw rate is the kinematic bicycle v*tan(delta)/L with the sent command.

Two estimates:
  1. cross-correlation: the shift of the commanded yaw rate that best matches the measured one
  2. step response (keyboard steps only): time from each steering step to 20% / 50% of the
     yaw-rate change (20% ~ dead time, 50% ~ dead time + half the rise)

usage (after tools/bag_extract.py):
    python3 tools/steer_delay.py ~/bags/steer_<time>_extracted                  # tools/steer_step_test.py run (/drive)
    python3 tools/steer_delay.py logs/<bag>_extracted --mode manual             # keyboard steps (/teleop)
  A policy run (agent_node) also works with --mode auto, but it is closed loop: biased short.
'''
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial import cKDTree

WHEELBASE = 0.34        # [m] configs/task/dynamic.yaml lf+lr
STEER_CMD_GAIN = 1.031  # command = wheel angle * gain (configs/task/f1tenth.yaml)


def scan_points(r, ang, max_r=8.0):
    m = np.isfinite(r) & (r > 0.05) & (r < max_r)
    return np.c_[r[m] * np.cos(ang[m]), r[m] * np.sin(ang[m])]


def icp_yaw(A, B, iters=30):
    '''rotation [rad] that maps scan B (later) onto scan A (earlier) = yaw change of the car'''
    tree = cKDTree(A)
    th, tr = 0.0, np.zeros(2)
    for k in range(iters):
        c, s = np.cos(th), np.sin(th)
        Bt = B @ np.array([[c, -s], [s, c]]).T + tr
        dist, j = tree.query(Bt)
        ok = dist < (0.3 if k < 10 else 0.1)
        if ok.sum() < 30:
            return np.nan
        P, Q = Bt[ok], A[j[ok]]
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        Rm = Vt.T @ U.T
        dth = np.arctan2(Rm[1, 0], Rm[0, 0])
        th += dth
        c, s = np.cos(dth), np.sin(dth)
        Rd = np.array([[c, -s], [s, c]])
        tr = Rd @ tr + (mq - Rd @ mp)
    return th


def segments(mask, min_len):
    '''[start, end) index ranges where mask is True for at least min_len samples'''
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dir')
    p.add_argument('--mode', default='auto', choices=['manual', 'auto'],
                   help='auto = commands on /drive (step test script or policy), manual = keyboard')
    p.add_argument('--min_speed', type=float, default=0.8, help='[m/s] ignore slower samples')
    p.add_argument('--step', type=float, default=0.2, help='[rad] command change counted as a step')
    a = p.parse_args()

    d = pd.read_csv(os.path.join(a.log_dir, 'driving.csv'))
    z = np.load(os.path.join(a.log_dir, 'scan.npz'))
    R, ts = z['ranges'], z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])

    # 100 Hz grid over the chosen mode while moving
    dt = 0.01
    grid = np.arange(d.t.min(), d.t.max(), dt)
    mode = d['mode'].values[np.clip(np.searchsorted(d.t.values, grid), 0, len(d) - 1)]
    cmd = np.interp(grid, d.t, d.cmd_steer)
    v = np.interp(grid, d.t, d.odom_speed)
    use = (mode == a.mode) & (v > a.min_speed)
    segs = segments(use, int(1.0 / dt))
    if not segs:
        print(f'[error] no {a.mode} driving above {a.min_speed} m/s for >= 1 s in this log')
        return
    print(f'[data] {len(segs)} segments, {sum(e - s for s, e in segs) * dt:.1f} s of {a.mode} driving '
          f'(mean speed {v[use].mean():.2f} m/s)')

    # measured yaw rate from scan matching, only around the used segments
    need = np.zeros(len(ts), bool)
    for s, e in segs:
        need |= (ts >= grid[s] - 0.5) & (ts <= grid[e - 1] + 0.5)
    idx = np.where(need)[0]
    yr_t, yr = [], []
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if i1 != i0 + 1:
            continue
        th = icp_yaw(scan_points(R[i0], ang), scan_points(R[i1], ang))
        if np.isfinite(th) and ts[i1] > ts[i0]:
            yr_t.append(0.5 * (ts[i0] + ts[i1]))
            yr.append(th / (ts[i1] - ts[i0]))
    yr_t, yr = np.array(yr_t), np.array(yr)
    meas = np.interp(grid, yr_t, yr)
    pred = v * np.tan(cmd / STEER_CMD_GAIN) / WHEELBASE
    b, af = butter(2, 4.0 / (0.5 / dt))
    measf, predf = filtfilt(b, af, meas), filtfilt(b, af, pred)

    # 1. cross-correlation over the used samples
    lags = np.arange(-0.10, 0.61, 0.01)
    corr = []
    for lag in lags:
        n = int(round(lag / dt))
        P, M, U = [], [], []
        for s, e in segs:
            if e - s <= abs(n) + 10:
                continue
            if n >= 0:
                P.append(predf[s:e - n]); M.append(measf[s + n:e])
            else:
                P.append(predf[s - n:e]); M.append(measf[s:e + n])
        corr.append(np.corrcoef(np.concatenate(P), np.concatenate(M))[0, 1])
    corr = np.array(corr)
    best = lags[np.argmax(corr)]
    print('\n[1] cross-correlation (commanded vs measured yaw rate)')
    for lag, c in zip(lags, corr):
        if round(lag * 100) % 5 == 0:
            print(f'    lag {lag:+.2f} s  corr {c:.3f}')
    print(f'    best lag {best:.2f} s (corr {corr.max():.3f}), measured/commanded yaw-rate gain '
          f'{np.std(measf[use]) / max(np.std(predf[use]), 1e-6):.2f}')
    print('    (valid for the step test; for a policy run it is biased short because the policy reacts to the car)')

    # 2. step responses (the command jumps by >= --step rad, then holds)
    print('\n[2] step responses')
    dcmd = np.abs(np.diff(cmd))
    steps = np.where((dcmd >= a.step) & use[1:])[0] + 1
    steps = steps[np.r_[True, np.diff(steps) > int(0.3 / dt)]] if len(steps) else steps
    measm = pd.Series(meas).rolling(5, center=True, min_periods=1).median().values
    rows = []
    for k in steps:
        pre = slice(k - int(0.15 / dt), k)
        post = slice(k + int(0.35 / dt), k + int(0.55 / dt))
        if pre.start < 0 or post.stop > len(grid) or not use[pre].all() or not use[k:post.stop].all():
            continue
        if np.abs(np.diff(cmd[k:post.stop])).max() > 0.05:      # command changed again too soon
            continue
        y0, y1 = measm[pre].mean(), measm[post].mean()
        if abs(y1 - y0) < 0.3:                                   # too small to time reliably
            continue
        frac = (measm[k:post.stop] - y0) / (y1 - y0)
        t20 = np.argmax(frac >= 0.2) * dt if (frac >= 0.2).any() else np.nan
        t50 = np.argmax(frac >= 0.5) * dt if (frac >= 0.5).any() else np.nan
        rows.append((grid[k], cmd[k - 1], cmd[k], v[k], y1 - y0, t20, t50))
    if rows:
        print('    t[s]     cmd from -> to   speed  d(yaw rate)   20%     50%')
        for r in rows:
            print(f'    {r[0]:7.2f}  {r[1]:+.2f} -> {r[2]:+.2f}   {r[3]:.2f}   {r[4]:+.2f} rad/s    {r[5]:.2f}s   {r[6]:.2f}s')
        t20s, t50s = np.array([r[5] for r in rows]), np.array([r[6] for r in rows])
        print(f'    {len(rows)} steps: 20% median {np.nanmedian(t20s):.2f} s, 50% median {np.nanmedian(t50s):.2f} s')
    else:
        print('    no usable steps (needs steering holds >= 0.6 s while driving above --min_speed)')

    print('\n[compare] sim2real.steer_delay in configs/task/f1tenth.yaml: [0.10, 0.30], nominal 0.25 s')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
'''
Fit the sim's tire parameters to real driving: friction mu and a cornering-stiffness scale k
(C_Sf, C_Sr = k * dynamic.yaml values), for the single-track model in
src/f1tenth/installation/f1tenth_gym/envs/dynamic_models.py (saturating tires since 11/23).

Real yaw rate: lidar scan matching (tools/steer_delay.py icp_yaw), 4 Hz low-pass.
Model input: the steering command actually sent (/ackermann_cmd), turned into a wheel angle with the
per-side full-lock calibration (STEER_CMD_GAIN_*), delayed by --delay and rate-limited like the sim;
speed = measured (odom, median-filtered against stale VESC packets in pre-6255ff3 logs).
Each driving segment is cut into WINDOW-second pieces; each piece starts from the measured yaw rate.

usage:
    python3 tools/fit_tires.py logs/run_1123_1350_extracted logs/run_1123_1402_extracted ...
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'f1tenth', 'installation'))
from steer_delay import scan_points, icp_yaw, segments  # noqa: E402
from f1tenth_gym.envs.dynamic_models import vehicle_dynamics_st  # noqa: E402

# car (configs/task/dynamic.yaml) and today's steering calibration (configs/task/f1tenth.yaml)
CAR = dict(lf=0.19, lr=0.15, h=0.074, m=3.74, I=0.04712)
C_SF, C_SR = 4.718, 5.4562
MAX_STEER = 0.41                 # [rad] measured full lock, both sides
STEER_CMD_GAIN_LEFT = 1.077      # command = angle * gain (left full lock 0.4415 -> 0.41 rad)
STEER_CMD_GAIN_RIGHT = 1.212     # (right full lock -0.497 -> 0.41 rad)
SV_MAX = 2.65                    # [rad/s] sim steering rate limit
DT = 0.005
WINDOW = 3.0                     # [s]


def wheel_angle(cmd):
    g = np.where(cmd > 0, STEER_CMD_GAIN_LEFT, STEER_CMD_GAIN_RIGHT)
    return np.clip(cmd / g, -MAX_STEER, MAX_STEER)


def load(log_dir, delay):
    d = pd.read_csv(os.path.join(log_dir, 'driving.csv'))
    z = np.load(os.path.join(log_dir, 'scan.npz'))
    R, ts = z['ranges'], z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
    # speed: rolling median removes stale VESC packets in older logs
    v = pd.Series(d.odom_speed.values).rolling(9, center=True, min_periods=1).median().values
    moving = (d['mode'] == 'auto').values & (v > 1.0)
    pieces = []
    for s, e in segments(moving, 40):
        t0, t1 = d.t.values[s], d.t.values[e - 1]
        k = np.where((ts >= t0) & (ts <= t1))[0]
        if len(k) < 60:
            continue
        yr = [0.0]
        for a, b in zip(k[:-1], k[1:]):
            th = icp_yaw(scan_points(R[a], ang), scan_points(R[b], ang))
            yr.append(th / (ts[b] - ts[a]) if np.isfinite(th) else np.nan)
        yr = pd.Series(yr).interpolate().bfill().values
        bb, aa = butter(2, 4.0 / 20.0)
        yr = filtfilt(bb, aa, yr)
        tk = ts[k]
        for w0 in np.arange(tk[0] + 0.5, tk[-1] - WINDOW, WINDOW):
            g = np.arange(w0, w0 + WINDOW, DT)
            pieces.append(dict(
                t=g,
                delta=wheel_angle(np.interp(g - delay, d.t, d.cmd_steer)),
                v=np.interp(g, d.t, v),
                yaw_meas=np.interp(g, tk, yr),
            ))
    return pieces


def simulate(p, mu, k):
    x = np.array([0.0, 0.0, p['delta'][0], p['v'][0], 0.0, p['yaw_meas'][0], 0.0])
    P = dict(mu=mu, C_Sf=C_SF * k, C_Sr=C_SR * k, s_min=-1.0, s_max=1.0, sv_min=-SV_MAX, sv_max=SV_MAX,
             v_switch=7.319, a_max=9.51, v_min=-8.0, v_max=8.0, **CAR)
    out = np.empty(len(p['t']))
    for i in range(len(p['t'])):
        sv = np.clip((p['delta'][i] - x[2]) / DT, -SV_MAX, SV_MAX)
        f = vehicle_dynamics_st(x, np.array([sv, 0.0]), **P)
        x = x + DT * f
        x[3] = p['v'][i]
        out[i] = x[5]
    return out


def residuals(params, pieces):
    mu, k = params
    r = [(simulate(p, mu, k) - p['yaw_meas'])[::10] for p in pieces]
    return np.concatenate(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_dirs', nargs='+')
    ap.add_argument('--delays', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.20])
    a = ap.parse_args()
    best = None
    for delay in a.delays:
        pieces = sum((load(ld, delay) for ld in a.log_dirs), [])
        fit = least_squares(residuals, x0=[0.7, 0.8], bounds=([0.2, 0.2], [2.0, 2.0]), args=(pieces,),
                            loss='soft_l1', f_scale=0.5, diff_step=0.02)
        rmse = np.sqrt(np.mean(fit.fun ** 2))
        base = np.sqrt(np.mean(residuals([1.0489, 1.0], pieces) ** 2))
        print(f'delay {delay:.2f} s: mu {fit.x[0]:.3f}, C_S scale {fit.x[1]:.3f} -> yaw-rate RMSE {rmse:.3f} rad/s '
              f'(current sim mu 1.049, scale 1: {base:.3f}), {len(pieces)} windows of {WINDOW:.0f} s')
        if best is None or rmse < best[0]:
            best = (rmse, delay, fit.x, pieces)
    rmse, delay, (mu, k), pieces = best
    print(f'\nbest: delay {delay:.2f} s, mu {mu:.3f} (peak lateral accel ~{mu * 9.81:.1f} m/s^2), '
          f'C_Sf {C_SF * k:.3f}, C_Sr {C_SR * k:.3f}')
    # how well it does at high lateral acceleration
    sim = np.concatenate([simulate(p, mu, k) for p in pieces])
    meas = np.concatenate([p['yaw_meas'] for p in pieces])
    v = np.concatenate([p['v'] for p in pieces])
    ay = np.abs(v * meas)
    for lo, hi in [(0, 3), (3, 5), (5, 8)]:
        m = (ay >= lo) & (ay < hi)
        if m.sum() > 50:
            print(f'  measured a_y {lo}-{hi} m/s^2: sim/real yaw rate {np.median(np.abs(sim[m]) / np.maximum(np.abs(meas[m]), 1e-3)):.2f} '
                  f'(n={m.sum()})')


if __name__ == '__main__':
    main()

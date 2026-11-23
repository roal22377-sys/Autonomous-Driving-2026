#!/usr/bin/env python3
'''
Analysis of tools/circle_test.py runs (bag extracted with tools/bag_extract.py).

For every constant-command step (steering, speed from /drive), skipping the first --settle seconds:
  - lidar trajectory from chained scan matching (tools/steer_delay.py icp_pose)
  - yaw rate w (slope of the accumulated yaw) and the lidar's circle radius (least-squares circle fit)
  - rear-axle radius R = sqrt(R_lidar^2 - LIDAR_X^2), true speed v = w * R
  - effective wheel angle atan(L / R): the kinematic angle that would give this circle
      at low lateral acceleration this is the real steering angle for the command
      as lateral acceleration grows, the circle grows (tires slide): the angle drops
  - lateral acceleration a_y = v * w (at the rear axle)
  - true speed / odom speed (speed_to_erpm_gain check)

Summary over all runs given:
  steering map: command -> real wheel angle (lowest a_y step per command), per side
  grip: highest a_y reached, and where the effective angle starts dropping
  speed: median true/odom ratio -> suggested speed_to_erpm_gain
  low speed: which commanded speeds the car actually held

usage:
    python3 tools/circle_analysis.py ~/bags/circle_L030_extracted ~/bags/circle_R030_extracted ...
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer_delay import scan_points, icp_pose  # noqa: E402

WHEELBASE = 0.34        # [m]
LIDAR_X = 0.32          # [m] rear axle -> lidar (front axle - 0.02, configs/task/dynamic.yaml)
STEER_CMD_GAIN = 1.031  # command = wheel angle * gain (current calibration)
VESC_GAIN = 4180.0      # speed_to_erpm_gain in vesc.yaml
STRIDE = 2              # scans between chained matches (20 Hz)


def fit_circle(x, y):
    '''least-squares (Kasa) circle fit -> center (cx, cy), radius'''
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, float(np.sqrt(c + cx ** 2 + cy ** 2))


def lidar_track(R, ang, idx):
    '''chained ICP over scan indices idx -> (yaw, x, y) of the lidar in the first scan's frame'''
    yaw, pos = 0.0, np.zeros(2)
    out = [(0.0, 0.0, 0.0)]
    prev = scan_points(R[idx[0]], ang)
    for i in idx[1:]:
        cur = scan_points(R[i], ang)
        th, tr, frac = icp_pose(prev, cur)
        if not np.isfinite(th) or frac < 0.5:
            return None
        c, s = np.cos(yaw), np.sin(yaw)
        pos = pos + np.array([[c, -s], [s, c]]) @ tr
        yaw += th
        out.append((yaw, pos[0], pos[1]))
        prev = cur
    return np.array(out)


def steps(d):
    '''constant (steer, speed) command windows on /drive: list of (t0, t1, steer, speed)'''
    a = d[d['mode'] == 'auto']
    if a.empty:
        return []
    key = (a.agent_steer.round(3).astype(str) + '/' + a.agent_speed.round(2).astype(str)).values
    out, s = [], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or key[i] != key[s] or a.t.values[i] - a.t.values[i - 1] > 0.3:
            out.append((a.t.values[s], a.t.values[i - 1], float(a.agent_steer.values[s]),
                        float(a.agent_speed.values[s])))
            s = i
    return out


def analyse(log_dir, settle):
    d = pd.read_csv(os.path.join(log_dir, 'driving.csv'))
    z = np.load(os.path.join(log_dir, 'scan.npz'))
    R, ts = z['ranges'], z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
    rows = []
    for t0, t1, steer, speed in steps(d):
        if speed <= 0 or t1 - (t0 + settle) < 1.0:
            continue
        idx = np.where((ts >= t0 + settle) & (ts <= t1))[0][::STRIDE]
        tr = lidar_track(R, ang, idx)
        row = dict(log=os.path.basename(os.path.normpath(log_dir)).replace('_extracted', ''),
                   steer=steer, speed_cmd=speed, dur=ts[idx[-1]] - ts[idx[0]])
        m = (d.t >= ts[idx[0]]) & (d.t <= ts[idx[-1]])
        row['v_odom'] = float(d.odom_speed[m].mean())
        if tr is None:
            rows.append(row)
            continue
        tt = ts[idx] - ts[idx[0]]
        w = float(np.polyfit(tt, tr[:, 0], 1)[0])
        swept = abs(tr[-1, 0])
        row.update(w=w, swept_deg=np.degrees(swept))
        if swept > np.radians(60):          # need a real arc for the radius
            _, _, r_l = fit_circle(tr[:, 1], tr[:, 2])
            r = np.sqrt(max(r_l ** 2 - LIDAR_X ** 2, 1e-4))
            v = abs(w) * r
            row.update(R=r, v_true=v, a_y=v * abs(w), angle_eff=np.arctan(WHEELBASE / r),
                       angle_cmd=abs(steer) / STEER_CMD_GAIN)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dirs', nargs='+')
    p.add_argument('--settle', type=float, default=1.0, help='[s] skip at the start of each step')
    a = p.parse_args()
    df = pd.concat([analyse(ld, a.settle) for ld in a.log_dirs], ignore_index=True)
    if df.empty:
        print('[error] no constant-command steps found (/drive in mode auto)')
        return
    pd.set_option('display.width', 200)
    cols = [c for c in ['log', 'steer', 'speed_cmd', 'v_odom', 'v_true', 'w', 'R', 'a_y', 'angle_cmd',
                        'angle_eff', 'swept_deg'] if c in df]
    print('== every step (angles in rad, a_y in m/s^2; v_true/R need >= 60 deg of arc)')
    print(df[cols].round(3).to_string(index=False))

    ok = df.dropna(subset=['v_true']) if 'v_true' in df else df.iloc[:0]
    if ok.empty:
        return
    print('\n== steering map (lowest lateral-accel step of each command; real = kinematic angle of the circle)')
    for steer, g in ok.groupby('steer'):
        b = g.sort_values('a_y').iloc[0]
        print(f'  command {steer:+.3f}: real angle {b.angle_eff:.3f} rad ({b.angle_eff / b.angle_cmd:.2f} x calibrated '
              f'{b.angle_cmd:.3f}) at a_y {b.a_y:.1f} m/s^2, {b.v_true:.1f} m/s')
    print('\n== grip: effective angle vs lateral accel (angle drops = tires sliding)')
    for steer, g in ok.groupby('steer'):
        g = g.sort_values('speed_cmd')
        base = g.sort_values('a_y').iloc[0].angle_eff
        line = ', '.join(f'{r.v_true:.1f}m/s a_y {r.a_y:.1f} -> {r.angle_eff / base:.2f}' for r in g.itertuples())
        print(f'  command {steer:+.3f}: {line}')
    print(f'  highest lateral accel reached: {ok.a_y.max():.1f} m/s^2 (sim: ~10 with mu 1.05)')
    r = (ok.v_true / ok.v_odom).median()
    print(f'\n== speed: true / odom = {r:.3f} (median of {len(ok)} steps) -> speed_to_erpm_gain '
          f'{VESC_GAIN:.0f} -> {VESC_GAIN / r:.0f}')
    low = df[df.speed_cmd < 2.0]
    if len(low):
        print('\n== low speed held? (commanded vs odom / true)')
        for rr in low.itertuples():
            vt = getattr(rr, 'v_true', np.nan)
            print(f'  {rr.speed_cmd:.1f} m/s: odom {rr.v_odom:.2f}, true {vt:.2f}')


if __name__ == '__main__':
    main()

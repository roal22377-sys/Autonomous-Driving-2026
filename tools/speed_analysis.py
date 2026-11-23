#!/usr/bin/env python3
'''
Analysis of tools/speed_test.py runs: true speed (lidar range to the wall straight ahead) vs /odom.

For every straight run (/drive steering 0, constant speed command), on the part where the speed is
steady (odom within 10 % of its median, wall farther than 1.5 m), the true speed is the slope of the
front range (median of the beams within +-2 deg) over time.

Also the car's acceleration and braking (11/24, for the simulator's a_max): from the odom speed (true
speed = odom x the factor printed above), the mean rate while the speed rises from 20 % to 80 % of the
command, and while it falls from 80 % to 20 % of the speed held after the command drops to 0 (the brake
before the wall). For the full motor
limit, run tools/speed_test.py with a high --speed (5-6 m/s) so the VESC is asked for everything it has.
usage:
    python3 tools/speed_analysis.py logs/run_<time>_extracted
'''
import argparse
import os

import numpy as np
import pandas as pd

VESC_GAIN = 4180.0      # speed_to_erpm_gain in vesc.yaml


def runs(d):
    a = d[(d['mode'] == 'auto') & (d.agent_speed > 0) & (d.agent_steer.abs() < 1e-3)]
    if a.empty:
        return []
    out, s = [], 0
    t = a.t.values
    for i in range(1, len(a) + 1):
        if i == len(a) or t[i] - t[i - 1] > 0.3:
            out.append((t[s], t[i - 1], float(a.agent_speed.values[s:i].max())))
            s = i
    return out


def ramp_rate(t, v, v_from, v_to):
    '''
        mean rate [m/s^2] while v goes from v_from to v_to (first crossings; v_from > v_to for braking):
        least-squares slope over that stretch, robust to odom noise (the steepest short window is not)
    '''
    if len(t) < 4:
        return np.nan
    up = v_to > v_from
    i0 = np.argmax(v >= v_from) if up else np.argmax(v <= v_from)
    rest = v[i0:]
    j = np.argmax(rest >= v_to) if up else np.argmax(rest <= v_to)
    if j < 3:
        return np.nan
    return float(np.polyfit(t[i0:i0 + j + 1], v[i0:i0 + j + 1], 1)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dirs', nargs='+')
    a = p.parse_args()
    rows, acc = [], []
    for ld in a.log_dirs:
        d = pd.read_csv(os.path.join(ld, 'driving.csv'))
        z = np.load(os.path.join(ld, 'scan.npz'))
        R, ts = z['ranges'], z['t']
        ang = np.degrees(float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment']))
        mid = np.abs(ang) < 2.0
        front = np.array([np.median(r[mid][np.isfinite(r[mid]) & (r[mid] > 0.05)]) if np.any(np.isfinite(r[mid]))
                          else np.nan for r in R])
        for t0, t1, v_cmd in runs(d):
            # acceleration: from the start of the run until the speed first reaches 90 % of the command;
            # braking: from the end of the run (command -> 0) for 2.5 s
            vfilt = d.odom_speed.rolling(5, center=True, min_periods=1).median()
            ma = (d.t >= t0) & (d.t <= t1)
            reach = d.t[ma & (vfilt >= 0.9 * v_cmd)]
            ta1 = float(reach.iloc[0]) if len(reach) else t1
            mu = (d.t >= t0) & (d.t <= ta1)
            up = ramp_rate(d.t[mu].values, vfilt[mu].values, 0.2 * v_cmd, 0.8 * v_cmd)
            mb = (d.t >= t1) & (d.t <= t1 + 2.5)
            v_b = float(vfilt[(d.t >= t1 - 0.2) & (d.t <= t1)].median()) if mb.any() else np.nan
            down = ramp_rate(d.t[mb].values, vfilt[mb].values, 0.8 * v_b, 0.2 * v_b) if np.isfinite(v_b) else np.nan
            acc.append(dict(log=os.path.basename(os.path.normpath(ld)).replace('_extracted', ''), cmd=v_cmd,
                            accel=up, brake=-down if np.isfinite(down) else np.nan))
            m = (d.t >= t0) & (d.t <= t1)
            vo = d.odom_speed[m].values
            tt = d.t[m].values
            med = np.median(vo[vo > 0.3]) if np.any(vo > 0.3) else np.nan
            steady = np.abs(vo - med) < 0.1 * med
            if steady.sum() < 20:
                continue
            ts0, ts1 = tt[steady][0], tt[steady][-1]
            k = (ts >= ts0) & (ts <= ts1) & np.isfinite(front) & (front > 1.5)
            if k.sum() < 15:
                continue
            slope, _ = np.polyfit(ts[k], front[k], 1)
            fit = np.polyval([slope, _], ts[k])
            rows.append(dict(log=os.path.basename(os.path.normpath(ld)).replace('_extracted', ''),
                             cmd=v_cmd, odom=float(np.mean(vo[steady])), true=-slope,
                             steady_s=ts1 - ts0, resid_cm=100 * float(np.std(front[k] - fit))))
    if not rows:
        print('[error] no straight constant-speed runs toward a wall found')
        return
    df = pd.DataFrame(rows)
    df['true/odom'] = df.true / df.odom
    df['true/cmd'] = df.true / df.cmd
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
    r = df['true/odom'].median()
    print(f'\ntrue speed = {r:.3f} x odom (median of {len(df)} runs)  ->  speed_to_erpm_gain {VESC_GAIN:.0f} -> '
          f'{VESC_GAIN / r:.0f}')
    print('(the gain sets both the odom reading and the command -> erpm conversion, so after the change '
          'commanded speed = true speed as well)')
    if acc:
        da = pd.DataFrame(acc)
        da[['accel', 'brake']] *= r                     # odom -> true speed
        print('\nacceleration / braking per run [m/s^2, true speed] (mean rate between 20 % and 80 % of the speed):')
        print(da.round(2).to_string(index=False))
        hi = da[da.cmd >= 4.0]
        src = hi if len(hi) else da
        print(f'-> simulator a_max (configs/task/f1tenth*.yaml sim2real.a_max): acceleration median '
              f'{src.accel.median():.2f}, max {src.accel.max():.2f} m/s^2 | braking median {src.brake.median():.2f}, '
              f'max {src.brake.max():.2f} m/s^2' + ('' if len(hi) else '  (no run commanded >= 4 m/s: may be below the motor limit)'))


if __name__ == '__main__':
    main()

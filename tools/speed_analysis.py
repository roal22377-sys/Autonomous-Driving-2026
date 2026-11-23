#!/usr/bin/env python3
'''
Analysis of tools/speed_test.py runs: true speed (lidar range to the wall straight ahead) vs /odom.

For every straight run (/drive steering 0, constant speed command), on the part where the speed is
steady (odom within 10 % of its median, wall farther than 1.5 m), the true speed is the slope of the
front range (median of the beams within +-2 deg) over time.

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dirs', nargs='+')
    a = p.parse_args()
    rows = []
    for ld in a.log_dirs:
        d = pd.read_csv(os.path.join(ld, 'driving.csv'))
        z = np.load(os.path.join(ld, 'scan.npz'))
        R, ts = z['ranges'], z['t']
        ang = np.degrees(float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment']))
        mid = np.abs(ang) < 2.0
        front = np.array([np.median(r[mid][np.isfinite(r[mid]) & (r[mid] > 0.05)]) if np.any(np.isfinite(r[mid]))
                          else np.nan for r in R])
        for t0, t1, v_cmd in runs(d):
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


if __name__ == '__main__':
    main()

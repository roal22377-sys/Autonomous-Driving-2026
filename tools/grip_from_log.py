#!/usr/bin/env python3
'''
Tire grip from an ordinary driving log (no circle test, no IMU): the car's heading from the map localization
(utils/localizer.py, 40 Hz) gives the yaw rate r, the localized path gives the speed v, so the lateral
acceleration is a_y = v r. The steering command (minus the steering delay, through the steering calibration)
gives the yaw rate the car would have WITHOUT sliding, r_kin = v tan(delta) / L. While the tires grip,
r / r_kin stays ~1; at the grip limit the car runs wide (understeer) or rotates more (oversteer) and the ratio
leaves 1. So:
    grip lower bound = the highest lateral acceleration reached while the ratio stayed ~1   (always available)
    grip estimate    = the lateral acceleration where the ratio leaves 1                     (only if the log got there)
The steering calibration is re-fitted per side from the low-acceleration part of the same log.

    python3 tools/grip_from_log.py LOG MAP --track COURSE.npz [--t0 --t1] [--delay 0.12]

LOG: logs/<run>_extracted (scan.npz + driving.csv with cmd_steer), MAP: maps/<run> of the same course.
Use logs with the speed cap raised step by step (corners get closer to the limit). Mu = grip / 9.81.
'''
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, 'tools'), os.path.join(REPO, 'src/f1tenth')]


def smooth(x, n):
    k = np.ones(2 * n + 1) / (2 * n + 1)
    xp = np.pad(x, n, mode='edge')
    return np.convolve(xp, k, mode='valid')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('map')
    ap.add_argument('--track', required=True)
    ap.add_argument('--t0', type=float, default=-np.inf)
    ap.add_argument('--t1', type=float, default=np.inf)
    ap.add_argument('--laser_x', type=float, default=0.13)
    ap.add_argument('--delay', type=float, default=0.12, help='[s] steering command -> wheel (11/24: 0.12)')
    ap.add_argument('--gain_left', type=float, default=0.866, help='command / wheel angle, left (11/24 calibration)')
    ap.add_argument('--gain_right', type=float, default=1.012)
    ap.add_argument('--wheelbase', type=float, default=0.34)
    ap.add_argument('--smooth', type=int, default=6, help='heading smoothing, samples each side (40 Hz: 6 = 0.33 s)')
    a = ap.parse_args()
    import pandas as pd
    import loc_test as LT
    from utils.localizer import ScanLocalizer, course_candidates
    R, ts, ang, od = LT.load_log(a.log, a.t0, a.t1)
    loc = ScanLocalizer(a.map, laser_x=a.laser_x)
    p0 = loc.global_init(R[0], ang, course_candidates(a.track), od[0])[0]
    if p0 is None or not loc.valid:
        raise SystemExit('no start pose: give --t0 where the car stands on the course')
    P, V = [], []
    for k in range(len(ts)):
        p, v, _ = loc.step(R[k], ang, od[k])
        P.append(p), V.append(v)
    P, V = np.array(P), np.array(V)
    n = a.smooth
    d = pd.read_csv(f'{a.log}/driving.csv')
    # speed: the wheel odometry (calibrated speed_to_erpm_gain), not the derivative of the localized position
    # (a few cm of localization jitter became 8.9 m/s on a 4 m/s-capped run, first version 11/26)
    do = d[np.isfinite(d['odom_speed'])]
    v = smooth(np.interp(ts, do['t'], do['odom_speed']), 2)
    # yaw rate: slope of the localized heading over a +-n window (least squares), jumps of the localization excluded
    psi = np.unwrap(P[:, 2])
    r = np.full(len(ts), np.nan)
    for i in range(n, len(ts) - n):
        tt, pp = ts[i - n:i + n + 1], psi[i - n:i + n + 1]
        r[i] = np.polyfit(tt - tt.mean(), pp, 1)[0]
    jump = np.r_[0, np.hypot(*np.diff(P[:, :2], axis=0).T)] > np.maximum(v, 0.5) * np.gradient(ts) * 2 + 0.05
    for i in np.nonzero(jump)[0]:
        r[max(0, i - n):i + n + 1] = np.nan
    ay = v * r
    ax = np.gradient(smooth(v, n)) / np.gradient(ts)
    d = d[np.isfinite(d['cmd_steer'])]
    cmd = np.interp(ts - a.delay, d['t'], d['cmd_steer'])
    delta = np.where(cmd > 0, cmd / a.gain_left, cmd / a.gain_right)
    r_kin = v * np.tan(delta) / a.wheelbase
    ok = V & (v > 1.0) & (np.abs(delta) > 0.04) & np.isfinite(r)
    side = np.sign(delta)
    fin = V & np.isfinite(ay)
    print(f'{len(ts)} scans, localization valid {V.mean()*100:.0f} %, speed max {v[V].max():.2f} m/s, '
          f'|a_y| p99 {np.percentile(np.abs(ay[fin]), 99):.2f} m/s^2, max {np.abs(ay[fin]).max():.2f}')
    for s, lab in ((1, 'left'), (-1, 'right')):
        m = ok & (side == s)
        low = m & (np.abs(ay) < 2.5)
        if low.sum() < 30:
            print(f'{lab}: too few turning samples ({m.sum()})')
            continue
        k = float(np.median(r[low] / r_kin[low]))
        ratio = r / (k * r_kin)
        print(f'{lab} turns ({m.sum()} samples): steering calibration factor {k:.2f} (wheel turns {k:.2f}x the calibrated angle)')
        if not 0.7 <= k <= 1.3:
            print('   [warn] the steering -> yaw-rate model does not fit this log (factor far from 1, and the ratio drifts '
                  'with the acceleration): the sliding test below is NOT reliable; use the lower bound and the plateau')
        rows, lower = [], 0.0
        for lo in np.arange(0, 12, 1.0):
            b = m & (np.abs(ay) >= lo) & (np.abs(ay) < lo + 1)
            if b.sum() < 8:
                continue
            med = float(np.median(ratio[b]))
            rows.append(f'{lo:.0f}-{lo+1:.0f} m/s^2: {b.sum():4d} samples, yaw-rate ratio {med:.2f}')
            if 0.85 <= med <= 1.2:
                lower = max(lower, float(np.percentile(np.abs(ay[b]), 90)))
        for row in rows:
            print('   ' + row)
        slip = [lo for lo in np.arange(0, 12, 1.0) if (m & (np.abs(ay) >= lo) & (np.abs(ay) < lo + 1)).sum() >= 8
                and not 0.85 <= float(np.median(ratio[m & (np.abs(ay) >= lo) & (np.abs(ay) < lo + 1)])) <= 1.2]
        print(f'   -> grip >= {lower:.1f} m/s^2 (mu >= {lower/9.81:.2f}) without sliding'
              + (f'; the ratio leaves 1 from {min(slip):.0f} m/s^2 (mu ~ {min(slip)/9.81:.2f})' if slip else
                 '; no sliding seen: drive faster corners for the limit'))
    # plateau test (no steering model needed): the highest lateral acceleration per 10 s; if it stops growing while
    # the speed cap is raised from run to run, that plateau is the grip
    seg = ((ts - ts[0]) // 10).astype(int)
    print('per 10 s: ' + ', '.join(f'{ts[0] + 10*i:.0f}s v {np.nanmax(np.where(seg == i, v, np.nan)):.1f} '
                                    f'a_y {np.nanpercentile(np.abs(ay[(seg == i) & fin]), 98):.1f}'
                                    for i in range(seg.max() + 1) if ((seg == i) & fin).sum() > 20))
    print(f'=> LOWER BOUND: |a_y| p98 {np.percentile(np.abs(ay[fin]), 98):.2f} m/s^2 = mu >= '
          f'{np.percentile(np.abs(ay[fin]), 98)/9.81:.2f} reached without a crash (not the limit unless it plateaus)')
    comb = np.hypot(ax, ay)
    print(f'combined |a| (friction circle) p99 {np.percentile(comb[fin & (v > 1)], 99):.2f} m/s^2')


if __name__ == '__main__':
    main()

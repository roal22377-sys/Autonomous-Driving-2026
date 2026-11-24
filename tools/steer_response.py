'''
Real-car steering response from a driving log (no extra test needed): the actual yaw rate (from lidar scan
matching, ICP) vs the yaw rate the command should give, v * tan(steer) / wheelbase.
  - delay: the command shift (0-0.4 s) with the best correlation
  - gain : left / right separately, actual / expected yaw rate (|steer| 0.05-0.35 rad, speed > 1 m/s),
           once with the scan-matching speed and once with the odom speed
  ICP speed / odom speed far from 1 -> check speed_to_erpm_gain (tools/drive_session.sh --speed).
  The same measure in the simulator (IL 11 at a 3 m/s cap, nominal settings, 11/24): delay 0.125 s, gain 0.52 both sides.
usage (on a PC, needs numpy, pandas, scipy):
    python3 tools/steer_response.py logs/<run>_extracted <t0> <t1> [<t0> <t1> ...]   # autonomous spans before any contact
'''
import sys
import numpy as np
import pandas as pd

log = sys.argv[1]
spans = [(float(a), float(b)) for a, b in zip(sys.argv[2::2], sys.argv[3::2])]
WB = 0.34
z = np.load(f'{log}/scan.npz')
R, ts = z['ranges'], z['t']
ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
blocked = np.nanpercentile(np.where(np.isfinite(R) & (R > 0.05) & (R < 60), R, 30.0), 90, axis=0) < 0.35
d = pd.read_csv(f'{log}/driving.csv')
dr = pd.read_csv(f'{log}/drive.csv')
for c in ('x', 'y', 'yaw'):
    d[c] = d[c].interpolate().bfill().ffill()
from scipy.spatial import cKDTree


def pts(i, rmax=8.0):
    r = R[i]
    ok = np.isfinite(r) & (r > 0.1) & (r < rmax) & ~blocked
    return np.c_[r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])]


def tf(p, x, y, th):
    c, s = np.cos(th), np.sin(th)
    return p @ np.array([[c, s], [-s, c]]) + [x, y]


def icp(src, tree, dst, x, y, th, it=25):
    for _ in range(it):
        p = tf(src, x, y, th)
        dist, j = tree.query(p)
        ok = dist < 0.3
        if ok.sum() < 30:
            break
        a, b = p[ok], dst[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[1] *= -1
            Rm = Vt.T @ U.T
        x, y = np.array([x, y]) @ Rm.T + (mb - ma @ Rm.T)
        th += np.arctan2(Rm[1, 0], Rm[0, 0])
    return x, y, th


rows = []
for t0, t1 in spans:
    idx = np.flatnonzero((ts >= t0) & (ts <= t1))
    poses, mp = [(0.0, 0.0, 0.0)], [pts(idx[0])]
    j0 = np.searchsorted(d.t.values, ts[idx[0]])
    prev = (d.x.values[j0], d.y.values[j0], d.yaw.values[j0])
    for i in idx[1:]:
        j = np.searchsorted(d.t.values, ts[i]).clip(0, len(d) - 1)
        od = (d.x.values[j], d.y.values[j], d.yaw.values[j])
        dx, dy = od[0] - prev[0], od[1] - prev[1]
        c, s = np.cos(-prev[2]), np.sin(-prev[2])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        x0, y0, th0 = poses[-1]
        g = (x0 + np.cos(th0) * lx - np.sin(th0) * ly, y0 + np.sin(th0) * lx + np.cos(th0) * ly, th0 + od[2] - prev[2])
        ref = np.vstack(mp[-20:])
        pose = icp(pts(i), cKDTree(ref), ref, *g)
        poses.append(pose)
        mp.append(tf(pts(i), *pose))
        prev = od
    P = np.array(poses)
    tt = ts[idx]
    # smooth, then rates (0.1 s central differences)
    k = 4
    th = np.unwrap(P[:, 2])
    yaw_rate = (th[2 * k:] - th[:-2 * k]) / (tt[2 * k:] - tt[:-2 * k])
    v = np.hypot(P[2 * k:, 0] - P[:-2 * k, 0], P[2 * k:, 1] - P[:-2 * k, 1]) / (tt[2 * k:] - tt[:-2 * k])
    tm = tt[k:-k]
    rows.append((tm, yaw_rate, v))
tm = np.concatenate([r[0] for r in rows]); yr = np.concatenate([r[1] for r in rows]); v = np.concatenate([r[2] for r in rows])
best = None
for lag in np.arange(0.0, 0.41, 0.025):
    st = np.interp(tm - lag, dr.t.values, dr['drive.steering_angle'].values)
    exp = v * np.tan(st) / WB
    ok = v > 1.0
    cc = np.corrcoef(exp[ok], yr[ok])[0, 1]
    print(f'  delay {lag:.3f} s: corr {cc:.3f}')
    if best is None or cc > best[1]:
        best = (lag, cc)
lag = best[0]
st = np.interp(tm - lag, dr.t.values, dr['drive.steering_angle'].values)
exp = v * np.tan(st) / WB
for name, sel in (('left (+)', st > 0.05), ('right (-)', st < -0.05)):
    ok = sel & (v > 1.0) & (np.abs(st) < 0.35)
    gain = np.sum(yr[ok] * exp[ok]) / np.sum(exp[ok] ** 2)       # least squares through 0
    print(f'{name}: n {ok.sum()}, actual/expected yaw rate {gain:.2f}')
print(f'best delay {lag:.3f} s (corr {best[1]:.3f}), samples {len(tm)}, mean speed {v[v > 1].mean():.2f} m/s')

# same with the odom speed (ICP heading is well constrained, ICP speed along corridors less so)
vo = np.interp(tm, d.t.values, d.odom_speed.fillna(0).values)
print(f'ICP speed / odom speed (moving): {np.median(v[vo > 1] / vo[vo > 1]):.2f}')
for lag2 in (0.1,):
    st = np.interp(tm - lag2, dr.t.values, dr['drive.steering_angle'].values)
    exp = vo * np.tan(st) / WB
    for name, sel in (('left (+)', st > 0.05), ('right (-)', st < -0.05)):
        ok = sel & (vo > 1.0) & (np.abs(st) < 0.35)
        gain = np.sum(yr[ok] * exp[ok]) / np.sum(exp[ok] ** 2)
        print(f'[odom speed] {name}: actual/expected yaw rate {gain:.2f}')

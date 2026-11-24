#!/usr/bin/env python3
'''
Detailed look at one autonomous segment of a calibrated drive (odom speed = true speed, gain 3100):
where the policy loses time or margin even when it does not touch a wall.

  pose      : lidar scan matching against everything mapped so far (as tools/log_map.py)
  per frame : true speed (odom), commanded speed, steering command, yaw rate and lateral acceleration
              (speed x yaw rate), closest wall left / right / front, distance to the corridor center
  laps      : the start point is passed again (within 0.8 m, heading within 45 deg, >= 5 s later)

usage (on a PC, numpy / pandas / scipy / matplotlib):
    python3 tools/drive_analysis.py logs/<run>_extracted <t_start> <t_end> [out_dir]
writes <out_dir>/drive_analysis.png and prints a summary (default out_dir: <log>/analysis)
'''
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(log, 'analysis')
os.makedirs(out, exist_ok=True)
z = np.load(f'{log}/scan.npz')
R, ts = z['ranges'], z['t']
ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
Rv = np.where(np.isfinite(R) & (R > 0.05) & (R < 60), R, np.nan)
blocked = np.nanpercentile(np.where(np.isfinite(Rv), Rv, 30.0), 90, axis=0) < 0.35
d = pd.read_csv(f'{log}/driving.csv')
for c in ('x', 'y', 'yaw'):
    d[c] = d[c].interpolate().bfill().ffill()


def pts(i, rmax=8.0):
    r = R[i]
    ok = np.isfinite(r) & (r > 0.1) & (r < rmax) & ~blocked
    return np.c_[r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])]


def tf(p, x, y, th):
    c, s = np.cos(th), np.sin(th)
    return p @ np.array([[c, s], [-s, c]]) + [x, y]


def icp(src, tree, dst, x, y, th, it=20):
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


# ---------------- poses (every scan, 40 Hz)
idx = np.flatnonzero((ts >= t0) & (ts <= t1))
gmap, poses, mp = {}, [(0.0, 0.0, 0.0)], [pts(idx[0])]
for q in mp[0]:
    gmap.setdefault((int(q[0] // .05), int(q[1] // .05)), q)
j0 = np.searchsorted(d.t.values, ts[idx[0]])
prev = (d.x.values[j0], d.y.values[j0], d.yaw.values[j0])
gt, gp, since = None, None, 99
for i in idx[1:]:
    j = np.searchsorted(d.t.values, ts[i]).clip(0, len(d) - 1)
    od = (d.x.values[j], d.y.values[j], d.yaw.values[j])
    dx, dy = od[0] - prev[0], od[1] - prev[1]
    c, s = np.cos(-prev[2]), np.sin(-prev[2])
    lx, ly = c * dx - s * dy, s * dx + c * dy
    x0, y0, th0 = poses[-1]
    g = (x0 + np.cos(th0) * lx - np.sin(th0) * ly, y0 + np.sin(th0) * lx + np.cos(th0) * ly, th0 + od[2] - prev[2])
    src = pts(i)
    ref = np.vstack(mp[-15:])
    pose = icp(src, cKDTree(ref), ref, *g)
    if since >= 20:
        gp = np.array(list(gmap.values()))
        gt, since = cKDTree(gp), 0
    pose = icp(src, gt, gp, *pose, it=8)
    since += 1
    poses.append(pose)
    mp.append(tf(src, *pose))
    for q in mp[-1]:
        gmap.setdefault((int(q[0] // .05), int(q[1] // .05)), q)
    prev = od
P = np.array(poses)
tt = ts[idx]
th = np.unwrap(P[:, 2])

# ---------------- per-frame quantities
k = 4                                                     # 0.1 s each side
yaw_rate = np.gradient(th, tt)
yaw_rate = np.convolve(yaw_rate, np.ones(2 * k + 1) / (2 * k + 1), mode='same')
v = np.interp(tt, d.t.values, d.odom_speed.fillna(0).values)
vcmd = np.interp(tt, d.t.values, d.cmd_speed.ffill().fillna(0).values)
steer = np.interp(tt, d.t.values, d.cmd_steer.ffill().fillna(0).values)
a_lat = v * yaw_rate
left = np.nanmin(np.where((ang[None] > np.radians(60)) & (ang[None] < np.radians(120)) & ~blocked[None], Rv[idx], np.nan)
                 * np.abs(np.sin(ang))[None], axis=1)
right = np.nanmin(np.where((ang[None] < np.radians(-60)) & (ang[None] > np.radians(-120)) & ~blocked[None], Rv[idx], np.nan)
                  * np.abs(np.sin(ang))[None], axis=1)
front = np.nanmin(np.where((np.abs(ang)[None] < np.radians(10)) & ~blocked[None], Rv[idx], np.nan), axis=1)
width = left + right
offset = (left - right) / 2.0                             # + = car is right of the corridor center
moving = v > 0.5

# ---------------- laps: back at the first pose
start = P[0]
laps = [0]
for n in range(1, len(P)):
    if tt[n] - tt[laps[-1]] < 5.0:
        continue
    dpos = np.hypot(*(P[n, :2] - start[:2]))
    dth = abs((P[n, 2] - start[2] + np.pi) % (2 * np.pi) - np.pi)
    if dpos < 0.8 and dth < np.radians(45):
        # the closest frame of this pass
        m = n
        while m + 1 < len(P) and np.hypot(*(P[m + 1, :2] - start[:2])) < np.hypot(*(P[m, :2] - start[:2])):
            m += 1
        laps.append(m)
lap_times = np.diff(tt[laps])

# ---------------- straights vs corners
straight = moving & (np.abs(np.convolve(steer, np.ones(41) / 41, mode='same')) < 0.05)
corner = moving & (np.abs(a_lat) > 1.5)
# weave on straights: steering minus its 1 s mean
steer_hp = steer - np.convolve(steer, np.ones(41) / 41, mode='same')
off_hp = offset - np.convolve(offset, np.ones(81) / 81, mode='same')

# ---------------- floor returns while braking (pitch): a straight line across the front 50 deg
fr = np.abs(ang) < np.radians(25)
X = (Rv[idx][:, fr] * np.cos(ang[fr]))
flat = np.array([np.nanmedian(x) if np.isfinite(x).sum() > 100 and (np.abs(x - np.nanmedian(x)) < 0.06).mean() > 0.85
                 and np.nanmedian(x) > 1.5 else np.nan for x in X])
accel = np.gradient(np.convolve(v, np.ones(9) / 9, mode='same'), tt)
braking = moving & (accel < -1.5)


def p(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float('nan')


print(f'== {os.path.basename(log)} {t0:.0f}-{t1:.0f} s: {len(P)} scans, {np.trapz(v, tt):.0f} m (odom)')
print(f'laps: {len(lap_times)} complete, times {np.round(lap_times, 2).tolist()} s '
      f'(spread {np.ptp(lap_times) if len(lap_times) > 1 else float("nan"):.2f} s)')
print(f'speed: mean {v[moving].mean():.2f}, p90 {p(v[moving], 90):.2f}, max {v.max():.2f} m/s | cap {np.nanmax(vcmd):.2f}; '
      f'commanded mean {vcmd[moving].mean():.2f}; true - command: mean {np.mean((v - vcmd)[moving]):+.2f}, '
      f'p95 {p((v - vcmd)[moving], 95):+.2f} m/s (overshoot)')
print(f'lateral accel |v*r|: p50 {p(np.abs(a_lat[moving]), 50):.2f}, p90 {p(np.abs(a_lat[moving]), 90):.2f}, '
      f'p99 {p(np.abs(a_lat[moving]), 99):.2f} m/s^2 (grip >= ~4.2 measured: corners use {p(np.abs(a_lat[corner]), 90) / 4.2 * 100:.0f} % of that at p90)')
print(f'corners (|a_lat| > 1.5): speed mean {v[corner].mean():.2f} m/s, {corner.mean() * 100:.0f} % of the time')
print(f'straights (|steer 1 s mean| < 0.05): speed mean {v[straight].mean():.2f} m/s, {straight.mean() * 100:.0f} % of the time, '
      f'steering weave rms {np.sqrt(np.mean(steer_hp[straight] ** 2)):.3f} rad, lateral wobble rms {np.sqrt(np.nanmean(off_hp[straight] ** 2)) * 100:.1f} cm')
print(f'walls: side min left {np.nanmin(left[moving]):.2f} / right {np.nanmin(right[moving]):.2f} m, side p5 {p(np.minimum(left, right)[moving], 5):.2f} m, '
      f'front min {np.nanmin(front[moving]):.2f} m | corridor width p50 {p(width[moving], 50):.2f} m, '
      f'offset from center p50 {p(offset[straight], 50) * 100:+.0f} cm on straights (+ = right of center)')
print(f'steering: |cmd| p99 {p(np.abs(steer[moving]), 99):.2f} rad, full lock (>0.39) {np.mean(np.abs(steer[moving]) > 0.39) * 100:.1f} %, '
      f'rate mean {np.mean(np.abs(np.diff(steer)) / np.diff(tt)):.2f} rad/s')
print(f'floor-like line across the front while braking: {np.isfinite(flat[braking]).mean() * 100:.0f} % of braking scans '
      f'(not braking: {np.isfinite(flat[moving & ~braking]).mean() * 100:.0f} %)')

# ---------------- figure
allp = np.array(list(gmap.values()))
fig = plt.figure(figsize=(16, 12))
for n, (val, lab, cmap, lim) in enumerate([(v, 'true speed [m/s]', 'viridis', (0, max(4.0, v.max()))),
                                            (np.abs(a_lat), '|lateral accel| [m/s^2]', 'magma', (0, 5)),
                                            (np.abs(steer_hp), 'steering weave |cmd - 1 s mean| [rad]', 'inferno', (0, 0.15))]):
    ax = fig.add_subplot(2, 3, n + 1)
    ax.scatter(allp[:, 0], allp[:, 1], s=0.1, c='0.6')
    sc = ax.scatter(P[:, 0], P[:, 1], c=val, cmap=cmap, s=2, vmin=lim[0], vmax=lim[1])
    plt.colorbar(sc, ax=ax, shrink=0.7, label=lab)
    ax.plot(*P[0, :2], 'r*', ms=12)
    ax.set_aspect('equal')
    ax.set_title(lab)
ax = fig.add_subplot(2, 1, 2)
ax.plot(tt - tt[0], v, label='true speed [m/s]')
ax.plot(tt - tt[0], vcmd, label='commanded speed [m/s]', alpha=0.7)
ax.plot(tt - tt[0], steer * 10, label='steering cmd x10 [rad]', alpha=0.6)
ax.plot(tt - tt[0], np.minimum(left, right), label='closest side wall [m]', alpha=0.7)
for l in laps:
    ax.axvline(tt[l] - tt[0], color='k', lw=0.5)
ax.set_xlabel('s'); ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
ax.set_title(f'laps (black lines): {np.round(lap_times, 2).tolist()} s')
fig.tight_layout()
fig.savefig(os.path.join(out, 'drive_analysis.png'), dpi=75)
print('saved', os.path.join(out, 'drive_analysis.png'))

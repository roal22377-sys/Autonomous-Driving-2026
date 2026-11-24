'''
실차 로그 구간을 LiDAR 스캔 매칭(ICP)으로 지도화: odom 방향 오차 없이 트랙 모양과 궤적을 봄.
usage: python icp_map.py <log_dir> <out_png> <t_start> <t_end> [t_mark ...]
'''
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log, out, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
marks = [float(x) for x in sys.argv[5:]]
z = np.load(f'{log}/scan.npz')
R, ts = z['ranges'], z['t']
ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
blocked = np.nanpercentile(np.where(np.isfinite(R) & (R > 0.05) & (R < 60), R, 30.0), 90, axis=0) < 0.35
d = pd.read_csv(f'{log}/driving.csv')
for c in ('x', 'y', 'yaw'):
    d[c] = d[c].interpolate().bfill().ffill()


def pts(i, rmax=8.0):
    r = R[i]
    ok = np.isfinite(r) & (r > 0.1) & (r < rmax) & ~blocked
    return np.c_[r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])]


def odom(t):
    j = np.searchsorted(d.t.values, t).clip(0, len(d) - 1)
    return d.x.values[j], d.y.values[j], d.yaw.values[j]


def tf(p, x, y, th):
    c, s = np.cos(th), np.sin(th)
    return p @ np.array([[c, s], [-s, c]]) + [x, y]


def icp(src, dst_tree, dst, x, y, th, it=25):
    for _ in range(it):
        p = tf(src, x, y, th)
        dist, j = dst_tree.query(p)
        ok = dist < 0.3
        if ok.sum() < 30:
            break
        a, b = p[ok], dst[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        H = (a - ma).T @ (b - mb)
        U, _, Vt = np.linalg.svd(H)
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[1] *= -1
            Rm = Vt.T @ U.T
        dth = np.arctan2(Rm[1, 0], Rm[0, 0])
        tvec = mb - ma @ Rm.T
        # compose: new pose = R * old + t
        x, y = np.array([x, y]) @ Rm.T + tvec
        th += dth
    return x, y, th


idx = np.flatnonzero((ts >= t0) & (ts <= t1))[::2]
VOX = 0.05                       # [m] global map grid
gmap = {}                        # voxel -> point: everything mapped so far (revisits re-anchor the pose: less drift)


def add_to_map(p):
    for q in p:
        gmap.setdefault((int(q[0] // VOX), int(q[1] // VOX)), q)


poses = [(0.0, 0.0, 0.0)]
mp = [pts(idx[0])]
add_to_map(mp[0])
gtree, gpts, since = None, None, 0
prev_od = odom(ts[idx[0]])
for i in idx[1:]:
    od = odom(ts[i])
    # odom increment in the previous car frame as the initial guess
    dx, dy = od[0] - prev_od[0], od[1] - prev_od[1]
    c, s = np.cos(-prev_od[2]), np.sin(-prev_od[2])
    lx, ly = c * dx - s * dy, s * dx + c * dy
    x0, y0, th0 = poses[-1]
    gx = x0 + np.cos(th0) * lx - np.sin(th0) * ly
    gy = y0 + np.sin(th0) * lx + np.cos(th0) * ly
    gth = th0 + (od[2] - prev_od[2])
    src = pts(i)
    ref = np.vstack(mp[-15:])
    pose = icp(src, cKDTree(ref), ref, gx, gy, gth)          # against the last scans
    if since >= 20 or gtree is None:                           # global map tree, rebuilt every 20 scans
        gpts = np.array(list(gmap.values()))
        gtree, since = cKDTree(gpts), 0
    pose = icp(src, gtree, gpts, *pose, it=10)                 # then against everything seen so far
    since += 1
    poses.append(pose)
    mp.append(tf(src, *pose))
    add_to_map(mp[-1])
    prev_od = od
poses = np.array(poses)
allp = np.array(list(gmap.values()))
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(allp[:, 0], allp[:, 1], s=0.2, c='k', alpha=0.3)
tt = ts[idx]
sc = ax.scatter(poses[:, 0], poses[:, 1], c=tt - t0, cmap='plasma', s=8, zorder=3)
plt.colorbar(sc, label='s', shrink=0.6)
LABEL_EVERY = max(10, len(poses) // 60)        # about 60 labels whatever the length
for k in range(0, len(poses), LABEL_EVERY):
    x, y, th = poses[k]
    ax.arrow(x, y, 0.4 * np.cos(th), 0.4 * np.sin(th), head_width=0.12, color='b', zorder=4)
    j = np.searchsorted(d.t.values, tt[k]).clip(0, len(d) - 1)
    st = d.cmd_steer.values[j]
    if np.isfinite(st):
        ax.annotate(f'{tt[k]-t0:.1f}s {st:+.2f}', (x, y), fontsize=7, color='purple')
for m in marks:
    k = np.searchsorted(tt, m).clip(0, len(tt) - 1)
    ax.plot(*poses[k, :2], 'rx', ms=14, mew=3)
ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title(f'{log.split("/")[-1]} t={t0:.0f}-{t1:.0f}s (scan-matched map; labels: time, steer cmd rad +=left)')
fig.savefig(out, dpi=85, bbox_inches='tight')
print('saved', out, len(poses), 'poses')

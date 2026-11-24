'''
실차 로그 run_1124_1344 (IL 11) 시각화: 자율 구간마다
  위: odom 좌표계에 LiDAR 점(구간 전체, 시간색) + 궤적 + 1초 간격 표시 + 접촉 지점
  아래: 접촉 2.0/1.0/0.5/0 s 전 LiDAR(차 좌표계, 위가 정면), 명령 조향 방향, 가장 먼 방향
usage: python3 log_view.py <log_dir> <out_dir>
'''
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
d = pd.read_csv(os.path.join(log, 'driving.csv'))
z = np.load(os.path.join(log, 'scan.npz'))
R, ts = z['ranges'], z['t']
ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
Rv = np.where(np.isfinite(R) & (R > 0.05), R, np.nan)
blocked = np.nanpercentile(np.where(np.isfinite(Rv), Rv, 30.0), 90, axis=0) < 0.35
close = (np.nan_to_num(Rv, nan=99) < 0.15) & ~blocked
contact_scan = close.sum(1) > 0.10 * (~blocked).sum()

auto = (d['mode'] == 'auto').values
edges = np.flatnonzero(np.diff(np.r_[0, auto.astype(int), 0]))
segs = list(zip(edges[::2], edges[1::2]))
# odom pose
for c in ('x', 'y', 'yaw'):
    d[c] = d[c].interpolate().bfill().ffill()

for k, (s, e) in enumerate(segs):
    t0, t1 = d.t.values[s], d.t.values[e - 1]
    sc = np.flatnonzero((ts >= t0) & (ts <= t1))
    cidx = [i for i in sc if contact_scan[i]]
    tc = ts[cidx[0]] if cidx else t1
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_axes([0.05, 0.42, 0.9, 0.55])
    for i in sc[::3]:
        if ts[i] > tc + 0.3:
            break
        j = np.searchsorted(d.t.values, ts[i]).clip(0, len(d) - 1)
        x, y, yaw = d.x.values[j], d.y.values[j], d.yaw.values[j]
        r = Rv[i]
        ok = np.isfinite(r) & ~blocked & (r < 10)
        px = x + r[ok] * np.cos(yaw + ang[ok])
        py = y + r[ok] * np.sin(yaw + ang[ok])
        ax.scatter(px, py, s=0.3, c=[plt.cm.viridis((ts[i] - t0) / max(tc - t0, 1e-3))], alpha=0.5)
    m = (d.t.values >= t0) & (d.t.values <= tc)
    ax.plot(d.x.values[m], d.y.values[m], 'r-', lw=2, label='trajectory (odom)')
    for tt in np.arange(t0, tc, 1.0):
        j = np.searchsorted(d.t.values, tt)
        ax.annotate(f'{tt - t0:.0f}s', (d.x.values[j], d.y.values[j]), fontsize=8, color='r')
    j = np.searchsorted(d.t.values, tc).clip(0, len(d) - 1)
    ax.plot(d.x.values[j], d.y.values[j], 'kx', ms=15, mew=3, label=f'contact at {tc - t0:.1f}s' if cidx else 'end')
    ax.set_aspect('equal'); ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    ax.set_title(f'segment {k + 1}: t={t0:.0f}-{tc:.0f}s (lidar points colored by time: dark=early)')
    for q, dt in enumerate((2.0, 1.0, 0.5, 0.0)):
        a2 = fig.add_axes([0.03 + q * 0.245, 0.03, 0.22, 0.34])
        i = np.searchsorted(ts, tc - dt).clip(0, len(ts) - 1)
        r = Rv[i]
        ok = np.isfinite(r) & ~blocked
        a2.scatter(-r[ok] * np.sin(ang[ok]), r[ok] * np.cos(ang[ok]), s=1, c='k')
        j = np.searchsorted(d.t.values, ts[i]).clip(0, len(d) - 1)
        st = d.cmd_steer.values[max(0, j - 5):j + 1]
        st = st[np.isfinite(st)]
        st = float(st[-1]) if len(st) else 0.0
        v = d.odom_speed.values[j]
        # far direction (smoothed ranges, front 180 deg)
        fr = np.abs(ang) < np.radians(90)
        rr = np.where(ok, r, 0.0)
        rs = np.convolve(rr, np.ones(15) / 15, mode='same')
        far = ang[fr][np.argmax(rs[fr])]
        a2.plot([0, -6 * np.sin(far)], [0, 6 * np.cos(far)], 'b--', lw=1, label='farthest')
        a2.plot([0, -3 * np.sin(st * 2)], [0, 3 * np.cos(st * 2)], 'r-', lw=2, label=f'steer {st:+.2f} rad (x2)')
        a2.plot(0, 0, 'r^', ms=10)
        a2.set_xlim(-6, 6); a2.set_ylim(-2, 10); a2.set_aspect('equal'); a2.grid(alpha=0.3)
        a2.set_title(f'{dt:.1f}s before: v={v:.1f} m/s', fontsize=9)
        a2.legend(fontsize=7, loc='lower left')
    fig.savefig(os.path.join(out, f'seg{k + 1}.png'), dpi=80)
    plt.close(fig)
    print(f'segment {k + 1}: {t0:.1f}-{t1:.1f}s, contact {"at %.1fs" % (tc - t0) if cidx else "none"}')

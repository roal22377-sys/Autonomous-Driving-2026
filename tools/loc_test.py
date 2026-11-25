#!/usr/bin/env python3
'''
Offline test of the car's localization (src/f1tenth/utils/localizer.py) on a recorded drive: the same code
agent_node runs, scan by scan, against a map from tools/map_offline.py.

    python3 tools/loc_test.py LOG MAP [--track COURSE.npz] [--truth CSV] [--t0 --t1] [--laser_x 0.13]

LOG    scan.npz + driving.csv (logs/<run>_extracted, or a tools/sim_map_test.py log)
MAP    map dir (map.yaml/map.pgm)
--track  course .npz in the map frame (tools/map_to_track.py): start by global search along it, and preview
         features (utils/preview.py) from the localized pose vs from the reference pose
--truth  reference poses t,x,y,yaw: the simulator's truth.csv (a sim log; then the map frame is aligned to it
         with MAP/poses.csv) or MAP/poses.csv of the same run (real logs: the map tool's own optimized poses)
--sim    the log is from tools/sim_map_test.py (laser 0.17 m ahead of the sim pose, truth.csv as reference);
         --map_truth: the truth.csv of the run the map was made from, when LOG is another run (e.g. a faster one)

Prints: accepted share, valid share, position / heading error (median, p95, max), the longest invalid stretch,
time per scan; preview feature errors; writes <out>/loc_test.png.
'''
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src/f1tenth'))
from utils.localizer import ScanLocalizer, course_candidates, wrap  # noqa: E402


def load_log(log, t0, t1):
    import pandas as pd
    z = np.load(f'{log}/scan.npz')
    R, ts = z['ranges'].astype(np.float32), z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
    d = pd.read_csv(f'{log}/driving.csv')
    d = d[np.isfinite(d['x']) & np.isfinite(d['y']) & np.isfinite(d['yaw'])]
    m = (ts >= t0) & (ts <= t1) & (ts >= d['t'].min()) & (ts <= d['t'].max())
    R, ts = R[m], ts[m]
    yaw = np.unwrap(d['yaw'].values)
    od = np.column_stack([np.interp(ts, d['t'], d['x']), np.interp(ts, d['t'], d['y']), wrap(np.interp(ts, d['t'], yaw))])
    # beams blocked by the car itself (always short): treated as no return, as in the map tool
    blocked = np.nanpercentile(np.where(np.isfinite(R) & (R > 0.05), R, 30.0), 90, axis=0) < 0.35
    R[:, blocked] = np.inf
    return R, ts, ang, od


def interp_pose(t, ref):
    yaw = np.unwrap(ref[:, 3])
    return np.column_stack([np.interp(t, ref[:, 0], ref[:, 1]), np.interp(t, ref[:, 0], ref[:, 2]),
                            wrap(np.interp(t, ref[:, 0], yaw))])


def align(map_poses, truth):
    '''rigid transform map frame -> truth frame from the map tool's keyframe poses (as tools/sim_map_test.py)'''
    tt = interp_pose(map_poses[:, 0], truth)
    A, B = map_poses[:, 1:3], tt[:, :2]
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[1] *= -1
        Rm = (U @ Vt).T
    tm = mb - Rm @ ma
    dth = np.arctan2(Rm[1, 0], Rm[0, 0])
    return lambda P: np.column_stack([P[:, :2] @ Rm.T + tm, wrap(P[:, 2] + dth)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('map')
    ap.add_argument('--track', default=None)
    ap.add_argument('--truth', default=None)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--map_truth', default=None, help='sim: truth.csv of the MAPPING run (frame alignment), default --truth')
    ap.add_argument('--t0', type=float, default=-np.inf)
    ap.add_argument('--t1', type=float, default=np.inf)
    ap.add_argument('--laser_x', type=float, default=None)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    laser_x = a.laser_x if a.laser_x is not None else (0.17 if a.sim else 0.13)
    R, ts, ang, od = load_log(a.log, a.t0, a.t1)
    loc = ScanLocalizer(a.map, laser_x=laser_x, stride=a.stride)
    ref, to_ref = None, (lambda P: P)
    if a.truth:
        ref = np.loadtxt(a.truth, delimiter=',', skiprows=1)[:, :4]
        if a.sim:
            mt = np.loadtxt(a.map_truth, delimiter=',', skiprows=1)[:, :4] if a.map_truth else ref
            to_ref = align(np.loadtxt(f'{a.map}/poses.csv', delimiter=',', skiprows=1)[:, :4], mt)
    # start
    t_start = time.time()
    if a.track:
        cand = course_candidates(a.track)
        p0, res0, sh0 = loc.global_init(R[0], ang, cand, od[0])
        print(f'[init] global search over {len(cand)} poses along the course: {p0.round(3) if p0 is not None else None}, '
              f'residual {res0*100:.1f} cm, matched {sh0*100:.0f}% ({time.time()-t_start:.2f} s)')
    elif ref is not None and not a.sim:
        loc.reset(interp_pose(ts[:1], ref)[0], od[0])
        print('[init] reference pose')
    else:
        raise SystemExit('need --track (global start) or --truth (real log: start at the reference pose)')
    P, V, A, T, INFO = [], [], [], [], []
    for k in range(len(ts)):
        t0 = time.perf_counter()
        p, v, info = loc.step(R[k], ang, od[k])
        T.append(time.perf_counter() - t0)
        P.append(p), V.append(v), A.append(info.get('accepted', False)), INFO.append(info)
    P, V, A, T = np.array(P), np.array(V), np.array(A), np.array(T)
    print(f'{len(ts)} scans {ts[0]:.1f}-{ts[-1]:.1f} s: accepted {A.mean()*100:.1f}%, valid {V.mean()*100:.1f}%, '
          f'time per scan median {np.median(T)*1e3:.1f} ms, p99 {np.percentile(T, 99)*1e3:.1f} ms (this PC)')
    inval = np.diff(np.r_[0, (~V).astype(int), 0])
    runs = (np.nonzero(inval == -1)[0] - np.nonzero(inval == 1)[0]) if (~V).any() else np.array([0])
    print(f'longest invalid stretch {runs.max()} scans ({runs.max()*np.median(np.diff(ts)):.2f} s), {int((inval == 1).sum())} stretches')
    out = a.out or os.path.join(a.map, 'loc_test')
    os.makedirs(out, exist_ok=True)
    if ref is not None:
        Q = interp_pose(ts, ref)
        Pm = to_ref(P)
        e = np.hypot(*(Pm[:, :2] - Q[:, :2]).T)
        eh = np.degrees(np.abs(wrap(Pm[:, 2] - Q[:, 2])))
        ev = e[V]
        print(f'position error (valid scans): median {np.median(ev)*100:.1f} cm, p95 {np.percentile(ev, 95)*100:.1f} cm, '
              f'max {ev.max()*100:.1f} cm | heading median {np.median(eh[V]):.2f} deg, p95 {np.percentile(eh[V], 95):.2f} deg')
        if a.track:
            from utils.preview import TrackPreview, get_preview_config
            z = np.load(a.track)
            cfg = get_preview_config({'track_preview': {'enabled': True}})
            tp = TrackPreview(np.asarray(z['raceline'], float), cfg)
            # preview from the localized pose vs from the reference pose (both in the map frame)
            inv = None
            if a.sim:
                # reference (truth frame) -> map frame: invert the alignment numerically via least squares on 3 points
                M = to_ref(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
                Rm = np.column_stack([M[1, :2] - M[0, :2], M[2, :2] - M[0, :2]])
                inv = lambda Qr: np.column_stack([(Qr[:, :2] - M[0, :2]) @ np.linalg.inv(Rm).T, wrap(Qr[:, 2] - M[0, 2])])
            Qm = inv(Q) if inv else Q
            idx = np.nonzero(V)[0][::5]
            F = np.array([tp.features(*P[i], None, True) for i in idx])
            G = np.array([tp.features(*Qm[i], None, True) for i in idx])
            d = np.abs(F - G)
            print(f'preview features (localized vs reference pose): e_y median {np.median(d[:, 0])*100:.1f} cm, '
                  f'e_psi median {np.degrees(np.median(d[:, 1])):.2f} deg, curvature@2m median {np.median(d[:, 2]):.3f} 1/m, '
                  f'speed@0 median {np.median(d[:, 10]):.2f} m/s')
        np.savetxt(f'{out}/loc.csv', np.column_stack([ts, P, V, A, e]), delimiter=',', header='t,x,y,yaw,valid,accepted,err',
                   comments='', fmt='%.4f')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2 if ref is not None else 1, figsize=(15, 6), squeeze=False)
    ax[0, 0].scatter(*loc.pts.T, s=0.2, c='0.5')
    ax[0, 0].plot(*P[:, :2].T, 'b-', lw=0.5, label='localized')
    ax[0, 0].scatter(*P[~V, :2].T, s=4, c='r', label='not valid')
    ax[0, 0].set_aspect('equal'), ax[0, 0].legend()
    if ref is not None:
        ax[0, 1].plot(ts, e * 100, 'b-', lw=0.6, label='position error [cm]')
        ax[0, 1].plot(ts, ~V * 20, 'r-', lw=0.6, label='not valid')
        ax[0, 1].set_xlabel('t [s]'), ax[0, 1].legend()
    fig.tight_layout()
    fig.savefig(f'{out}/loc_test.png', dpi=100)
    print(f'[saved] {out}/loc_test.png')


if __name__ == '__main__':
    main()

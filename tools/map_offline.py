#!/usr/bin/env python3
'''
Offline 2D SLAM for a recorded drive: scan matching + loop closure + pose-graph optimization -> occupancy map.

    python3 tools/map_offline.py logs/<run>_extracted maps/<name> [--t0 S --t1 S] [--res 0.05]

Input : the extracted bag (tools/bag_extract.py): scan.npz (all 1081 Hokuyo beams) and driving.csv (odom x, y, yaw).
Output: maps/<name>/map.pgm + map.yaml (ROS map_server format: 0 = wall, 254 = free, 205 = unknown),
        poses.csv (keyframe poses in the map frame), preview.png (odom-only vs optimized map, loop closures),
        report.txt (keyframes, loop closures, residuals).

Why not tools/log_map.py: it chains every scan onto the map built so far without ever closing the loop, so the
drift of a lap stays in the map (11/24 run_1124_1521: walls doubled 0.3-0.5 m apart). Here, every time the car
passes a place again, the two scans are matched and a "same place" constraint is added; the pose graph then spreads
the accumulated error over the whole lap and the walls line up (src/f1tenth/MAP_TOOL.md).

Steps: keyframes every 0.15 m / 5 deg of odom motion -> point-to-line ICP of each keyframe against a local submap
(odom delta as the initial guess) -> loop-closure candidates (same place, >= 15 s later) matched by ICP and
accepted only with a tight fit -> least-squares pose graph (scipy) -> second round of loop closures -> occupancy grid
by ray casting (log-odds) from the optimized poses.
'''
import argparse
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

LASER_X = 0.13            # [m] base_link -> laser (bringup_launch.py static transform)
R_MIN, R_MAX = 0.10, 10.0  # [m] valid ranges (Hokuyo UST-10LX)


# ------------------------------------------------------------------ geometry
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    '''pose a (x, y, th) followed by the relative pose b'''
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def relative(a, b):
    '''pose of b in the frame of a'''
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def transform(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def voxel(pts, size):
    if len(pts) == 0:
        return pts
    k = np.floor(pts / size).astype(np.int64)
    _, i = np.unique(k[:, 0] * 1_000_003 + k[:, 1], return_index=True)
    return pts[np.sort(i)]


# ------------------------------------------------------------------ ICP
def normals(pts, tree, k=14):
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]                                   # smallest eigenvalue -> normal
    flat = w[:, 0] / (w[:, 1] + 1e-9) < 0.3         # only well-defined lines (k=14: ~0.3 m of wall; 6 failed once
                                                    # a few scans thickened the wall to a 4-6 cm band; 0.2 dropped half
                                                    # of the rough walls of the generated tracks)
    return n, flat


def icp(src, tgt, tgt_tree, tgt_n, tgt_ok, init, iters=25, d_max=(0.6, 0.3, 0.15)):
    '''point-to-line ICP of src (sensor-frame points) onto tgt (world points with normals).
    returns pose, mean |residual| of the inliers [m], number of inliers'''
    pose = np.array(init, float)
    res, share = np.inf, 0.0
    for it in range(iters):
        dm = d_max[min(it * len(d_max) // iters, len(d_max) - 1)]
        P = transform(src, pose)
        d, j = tgt_tree.query(P, distance_upper_bound=dm)
        m = np.isfinite(d)
        m[m] = tgt_ok[j[m]]
        if m.sum() < 40:
            return pose, np.inf, 0
        p, q, n = P[m], tgt[j[m]], tgt_n[j[m]]
        r = np.einsum('ij,ij->i', p - q, n)
        keep = np.abs(r) <= np.quantile(np.abs(r), 0.85)         # trim the worst 15 %
        p, n, r = p[keep], n[keep], r[keep]
        # linearized about the current pose: dp = [dx, dy] + dth * perp(p - t)
        rel = p - pose[:2]
        J = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * -rel[:, 1] + n[:, 1] * rel[:, 0]])
        H = J.T @ J + np.diag([1e-3, 1e-3, 1e-3]) * len(r)          # damping keeps degenerate corridors near init
        dx = -np.linalg.solve(H, J.T @ r)
        pose = np.array([pose[0] + dx[0], pose[1] + dx[1], wrap(pose[2] + dx[2])])
        res, share = float(np.mean(np.abs(r))), int(keep.sum())
        Ht = J[:, :2].T @ J[:, :2]
        cons = float(np.linalg.eigvalsh(Ht)[0] / (np.trace(Ht) + 1e-9))    # 0 = slides along a corridor, 0.5 = fixed
        if np.abs(dx[:2]).max() < 1e-4 and abs(dx[2]) < 1e-4:
            break
    icp.constraint = cons
    return pose, res, share


class Submap:
    '''world points of the given keyframes, with normals and a KD-tree'''
    def __init__(self, clouds, poses, ids, vox=0.04):
        pts = np.concatenate([transform(clouds[i], poses[i]) for i in ids])
        self.pts = voxel(pts, vox)
        self.tree = cKDTree(self.pts)
        self.n, self.ok = normals(self.pts, self.tree)


# ------------------------------------------------------------------ pose graph
def distinct(cloud, k=10):
    '''how well a scan pins down the position: 0 = straight corridor (normals all parallel), 0.5 = corner'''
    if len(cloud) < 30:
        return 0.0
    n, ok = normals(cloud, cKDTree(cloud), k=k)
    n = n[ok]
    if len(n) < 20:
        return 0.0
    M = n.T @ n / len(n)
    return float(np.linalg.eigvalsh(M)[0])


def optimize(poses, edges, fix=0):
    '''edges: (a, b, z (rel pose of b in a), w_xy, w_th). Least squares over all poses but `fix`.'''
    N = len(poses)
    free = [i for i in range(N) if i != fix]
    col = {i: k for k, i in enumerate(free)}
    E = len(edges)
    A = np.array([e[0] for e in edges]); B = np.array([e[1] for e in edges])
    Z = np.array([e[2] for e in edges]); W = np.array([[e[3], e[3], e[4]] for e in edges])
    x0 = poses.copy()

    def unpack(v):
        P = x0.copy()
        P[free] = v.reshape(-1, 3)
        return P

    def resid(v):
        P = unpack(v)
        pa, pb = P[A], P[B]
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        r = np.column_stack([c * dx + s * dy - Z[:, 0], -s * dx + c * dy - Z[:, 1], wrap(pb[:, 2] - pa[:, 2] - Z[:, 2])])
        return (r * W).ravel()

    S = lil_matrix((3 * E, 3 * len(free)), dtype=np.int8)
    for e, (a, b) in enumerate(zip(A, B)):
        for node in (a, b):
            if node in col:
                S[3 * e:3 * e + 3, 3 * col[node]:3 * col[node] + 3] = 1
    # soft_l1: an occasional wrong loop closure is down-weighted instead of bending the whole map
    sol = least_squares(resid, x0[free].ravel(), jac_sparsity=S, method='trf', x_scale='jac', max_nfev=80, verbose=0,
                        loss='soft_l1', f_scale=2.0)
    return unpack(sol.x), float(np.sqrt(np.mean(sol.fun ** 2)))


# ------------------------------------------------------------------ occupancy grid
def occupancy(clouds_full, poses, res, pad=1.0, beam_stride=2):
    sensor = np.array([transform(np.array([[LASER_X, 0.0]]), p)[0] for p in poses])
    allpts = np.concatenate([transform(c, p) for c, p in zip(clouds_full, poses)])
    lo = np.minimum(allpts.min(0), sensor.min(0)) - pad
    hi = np.maximum(allpts.max(0), sensor.max(0)) + pad
    W, H = (np.ceil((hi - lo) / res).astype(int) + 1)
    hits = np.zeros(W * H, np.float32)
    passes = np.zeros(W * H, np.float32)
    for c, p, o in zip(clouds_full, poses, sensor):
        end = transform(c[::beam_stride], p)
        v = end - o
        L = np.linalg.norm(v, axis=1)
        n = np.maximum((L / (res * 0.8)).astype(int), 1)
        rep = np.repeat(np.arange(len(end)), n)
        frac = (np.arange(rep.size) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
        pts = o + v[rep] * frac[:, None]                       # free space along each beam (end excluded)
        ij = np.floor((pts - lo) / res).astype(int)
        np.add.at(passes, ij[:, 1] * W + ij[:, 0], 1.0)
        ij = np.floor((transform(c, p) - lo) / res).astype(int)
        np.add.at(hits, ij[:, 1] * W + ij[:, 0], 1.0)
    hits, passes = hits.reshape(H, W), passes.reshape(H, W)
    occ_p = hits / np.maximum(hits + passes, 1e-6)
    grid = np.full((H, W), 205, np.uint8)                       # unknown
    grid[(passes + hits) > 0] = 254                             # seen -> free unless hit
    grid[(hits >= 2) & (occ_p > 0.25)] = 0                      # wall
    return grid[::-1], lo                                       # image rows top = +y


def save_map(grid, lo, res, out):
    os.makedirs(out, exist_ok=True)
    H, W = grid.shape
    with open(f'{out}/map.pgm', 'wb') as f:
        f.write(f'P5\n{W} {H}\n255\n'.encode())
        f.write(grid.tobytes())
    with open(f'{out}/map.yaml', 'w') as f:
        f.write(f'image: map.pgm\nresolution: {res}\norigin: [{lo[0]:.4f}, {lo[1]:.4f}, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n')


# ------------------------------------------------------------------ main
def load(log, t0, t1):
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
    # beams blocked by the car itself (always short) are dropped
    blocked = np.nanpercentile(np.where(np.isfinite(R) & (R > 0.05), R, 30.0), 90, axis=0) < 0.35
    return R, ts, ang, od, blocked


def cloud(r, ang, blocked, rmax=R_MAX):
    ok = np.isfinite(r) & (r > R_MIN) & (r < rmax) & ~blocked
    return np.column_stack([LASER_X + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])


def set_laser_x(x):
    global LASER_X
    LASER_X = float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log'), ap.add_argument('out')
    ap.add_argument('--t0', type=float, default=0.0), ap.add_argument('--t1', type=float, default=1e9)
    ap.add_argument('--res', type=float, default=0.05, help='[m/cell] map resolution')
    ap.add_argument('--laser_x', type=float, default=LASER_X, help='[m] pose reference -> laser along x (car: 0.13 '
                    'base_link; simulator: 0.17 from the CG, tools/sim_map_test.py)')
    ap.add_argument('--kf_dist', type=float, default=0.15), ap.add_argument('--kf_ang', type=float, default=5.0)
    ap.add_argument('--loop_dist', type=float, default=1.5, help='[m] loop-closure search radius (after the drift is fixed)')
    ap.add_argument('--loop_drift', type=float, default=0.04,
                    help='[m/m] drift allowed per m driven between two passes (phase A search radius)')
    ap.add_argument('--loop_max', type=float, default=12.0, help='[m] phase A search radius cap')
    ap.add_argument('--loop_res', type=float, default=0.015, help='[m] loop closure: mean ICP residual below this')
    ap.add_argument('--loop_inliers', type=float, default=0.25, help='loop closure: share of the scan matched (>= 80 points)')
    ap.add_argument('--loop_prune', type=float, default=0.25,
                    help='[m] after optimizing, loop closures off by more than this are dropped')
    ap.add_argument('--distinct', type=float, default=0.12,
                    help='phase A uses keyframes whose wall normals point in several directions (corners): '
                         'smallest eigenvalue of the mean n n^T above this (0 straight corridor, 0.5 corner)')
    ap.add_argument('--loop_gap', type=float, default=15.0, help='[s] minimum time between the two passes')
    a = ap.parse_args()
    set_laser_x(a.laser_x)
    T0 = time.time()
    R, ts, ang, od, blocked = load(a.log, a.t0, a.t1)
    # keyframes on odom motion
    kf = [0]
    for i in range(1, len(ts)):
        rel = relative(od[kf[-1]], od[i])
        if np.hypot(rel[0], rel[1]) > a.kf_dist or abs(rel[2]) > np.radians(a.kf_ang):
            kf.append(i)
    kf = np.array(kf)
    clouds = [voxel(cloud(R[i], ang, blocked), 0.05) for i in kf]
    print(f'[map] {len(ts)} scans {ts[0]:.1f}-{ts[-1]:.1f} s -> {len(kf)} keyframes', flush=True)

    # 1) sequential scan matching against a local submap (initial guess: odom delta)
    poses = np.zeros((len(kf), 3))
    edges, seq_bad = [], 0
    for k in range(1, len(kf)):
        guess = compose(poses[k - 1], relative(od[kf[k - 1]], od[kf[k]]))
        ids = list(range(max(0, k - 12), k))
        sm = Submap(clouds, poses, ids)
        p, res, share = icp(clouds[k], sm.pts, sm.tree, sm.n, sm.ok, guess)
        if not (res < 0.04 and share >= 80) or np.hypot(*(p[:2] - guess[:2])) > 0.3:
            p, seq_bad = guess, seq_bad + 1                         # fall back to odom
            edges.append((k - 1, k, relative(od[kf[k - 1]], od[kf[k]]), 5.0, 5.0))
        else:
            edges.append((k - 1, k, relative(poses[k - 1], p), 20.0, 40.0))
        poses[k] = p
    seq_poses = poses.copy()
    odom_poses = np.array([compose(np.zeros(3), relative(od[kf[0]], od[i])) for i in kf])
    print(f'[map] sequential ICP done ({seq_bad} keyframes fell back to odom) {time.time()-T0:.0f}s', flush=True)

    # 2) loop closures + pose graph.
    # Phase A (big drift, 11/25 sim test: 280-460 m laps drifted > 1.5 m and found no closure at all): at distinctive
    # keyframes (corners; a straight corridor lets ICP slide along it), try "k is the same place as j" for earlier
    # distinctive keyframes j within the drift a pass that far back can have (loop_dist + loop_drift x distance
    # driven, <= loop_max); ICP starts from j's pose; keep the best tight, well-constrained fit. Then optimize.
    # Phase B (small residual drift): nearest keyframe within loop_dist, ICP from k's own pose, until none is new.
    s_kf = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
    dist_k = np.array([distinct(c) for c in clouds])
    dk, last = [], -1e9
    for i in np.where(dist_k > a.distinct)[0]:          # distinctive keyframes, >= 2 m apart along the path
        if s_kf[i] - last >= 2.0:
            dk.append(i)
            last = s_kf[i]
    dk = np.array(dk, int)
    loops, have = [], set()

    def try_close(j, k, guess, d_max):
        ids = [i for i in range(max(0, j - 6), min(len(kf), j + 7))]
        sm = Submap(clouds, poses, ids)
        p, res, n_in = icp(clouds[k], sm.pts, sm.tree, sm.n, sm.ok, guess, iters=40, d_max=d_max)
        return p, res, n_in, icp.constraint

    def good(res, n_in, n, cons, cons_min):
        # a true revisit matches to < 1 cm (sim test 11/25: 0.6-0.9 cm); share alone failed on rough walls
        return res < a.loop_res and n_in >= max(80, a.loop_inliers * n) and cons > cons_min

    def solve(poses, loops):
        '''optimize, then drop loop closures the optimized map disagrees with (> loop_prune m) and redo'''
        for _ in range(3):
            P, rms = optimize(poses, edges + loops)
            bad = [l for l in loops if np.hypot(*relative(relative(P[l[0]], P[l[1]]), l[2])[:2]) > a.loop_prune]
            if not bad:
                return P, rms, loops
            loops = [l for l in loops if l not in bad]
            print(f'[map] dropped {len(bad)} loop closure(s) the map disagrees with', flush=True)
        return optimize(poses, edges + loops) + (loops,)

    found_a = 0
    for k in dk:
        if ts[kf[k]] - ts[kf[0]] < a.loop_gap:
            continue
        dd = np.hypot(*(poses[dk, :2] - poses[k, :2]).T)
        ok = ((ts[kf[k]] - ts[kf[dk]] > a.loop_gap)
              & (dd <= np.minimum(a.loop_max, a.loop_dist + a.loop_drift * (s_kf[k] - s_kf[dk])))
              & (np.abs(wrap(poses[k, 2] - poses[dk, 2])) < np.radians(45)))
        cand = dk[ok][np.argsort(dd[ok])][:8]
        best = None
        for j in cand:
            guess = np.array([poses[j, 0], poses[j, 1], poses[j, 2] + wrap(0.0)])
            p, res, n_in, cons = try_close(j, k, guess, (1.0, 0.5, 0.25, 0.1))
            if good(res, n_in, len(clouds[k]), cons, 0.05) and (best is None or res < best[1]):
                best = (j, res, p)
        if best is not None and (best[0], k) not in have:
            j, _, p = best
            loops.append((j, k, relative(poses[j], p), 20.0, 40.0))
            have.add((j, k))
            found_a += 1
    if loops:
        poses, rms, loops = solve(poses, loops)
        have = {(l[0], l[1]) for l in loops}
    print(f'[map] phase A: {len(dk)} distinctive keyframes, +{found_a} loop closures {time.time()-T0:.0f}s', flush=True)
    for rnd in range(4):
        tree = cKDTree(poses[:, :2])
        found = 0
        for k in range(0, len(kf), 2):
            cand = [j for j in tree.query_ball_point(poses[k, :2], a.loop_dist)
                    if ts[kf[k]] - ts[kf[j]] > a.loop_gap and abs(wrap(poses[k, 2] - poses[j, 2])) < np.radians(60)]
            if not cand:
                continue
            j = min(cand, key=lambda c: np.hypot(*(poses[c, :2] - poses[k, :2])))
            if (j, k) in have:
                continue
            p, res, n_in, cons = try_close(j, k, poses[k], (1.0, 0.5, 0.2, 0.1))
            if good(res, n_in, len(clouds[k]), cons, 0.02) and np.hypot(*(p[:2] - poses[k, :2])) < 1.2:
                loops.append((j, k, relative(poses[j], p), 20.0, 40.0))
                have.add((j, k))
                found += 1
        if found == 0:
            break
        poses, rms, loops = solve(poses, loops)
        print(f'[map] phase B round {rnd + 1}: +{found} loop closures (total {len(loops)}), graph rms {rms:.3f} {time.time()-T0:.0f}s', flush=True)

    # 3) maps
    full = [cloud(R[i], ang, blocked) for i in kf]
    grid, lo = occupancy(full, poses, a.res)
    save_map(grid, lo, a.res, a.out)
    with open(f'{a.out}/poses.csv', 'w') as f:
        f.write('t,x,y,yaw\n')
        for i, p in zip(kf, poses):
            f.write(f'{ts[i]:.4f},{p[0]:.4f},{p[1]:.4f},{p[2]:.5f}\n')
    corr = np.hypot(*(poses[:, :2] - seq_poses[:, :2]).T)
    rep = (f'log {a.log} t {ts[0]:.1f}-{ts[-1]:.1f} s, {len(kf)} keyframes, path {np.sum(np.hypot(*np.diff(poses[:, :2], axis=0).T)):.1f} m\n'
           f'sequential ICP fell back to odom on {seq_bad} keyframes\n'
           f'loop closures {len(loops)}; correction of the sequential path by the pose graph: median {np.median(corr):.3f} m, max {corr.max():.3f} m\n'
           f'map {grid.shape[1]} x {grid.shape[0]} cells at {a.res} m, origin {lo.round(3).tolist()}\n')
    open(f'{a.out}/report.txt', 'w').write(rep)
    print(rep, end='')
    preview(odom_poses, seq_poses, poses, loops, clouds, grid, lo, a.res, f'{a.out}/preview.png', a.log)
    print(f'[done] {a.out}/map.pgm, map.yaml, poses.csv, preview.png ({time.time()-T0:.0f} s)')


def preview(odom_poses, seq_poses, poses, loops, clouds, grid, lo, res, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a_, P, name in ((ax[0], odom_poses, 'odom only'), (ax[1], seq_poses, 'scan matching, no loop closure')):
        pts = np.concatenate([transform(c, p) for c, p in zip(clouds[::2], P[::2])])
        a_.scatter(pts[:, 0], pts[:, 1], s=0.05, c='k', alpha=0.3)
        a_.plot(P[:, 0], P[:, 1], 'r-', lw=0.6)
        a_.set_title(name), a_.set_aspect('equal')
    H, W = grid.shape
    ax[2].imshow(grid, cmap='gray', extent=[lo[0], lo[0] + W * res, lo[1], lo[1] + H * res], vmin=0, vmax=255)
    ax[2].plot(poses[:, 0], poses[:, 1], 'r-', lw=0.6)
    for j, k, *_ in loops:
        ax[2].plot([poses[j, 0], poses[k, 0]], [poses[j, 1], poses[k, 1]], 'c-', lw=0.4)
    ax[2].set_title(f'loop closure + pose graph ({len(loops)} loops): map.pgm'), ax[2].set_aspect('equal')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()

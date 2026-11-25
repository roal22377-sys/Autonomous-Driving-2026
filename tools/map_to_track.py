#!/usr/bin/env python3
'''
Occupancy map of a real course -> simulator track (.npz in the utils/track_gen.py format), both directions.

    python3 tools/map_to_track.py maps/<name> --name Practice1127
        [--poses maps/<name>/poses.csv] [--max_half 3.0] [--out src/f1tenth/installation/maps_real]

Input : maps/<name>/map.yaml + map.pgm (tools/map_offline.py, or slam_toolbox's map_saver) and the car's path in
        that map frame (poses.csv from map_offline.py; one full lap is enough). The path gives the driving
        direction and the drivable corridor (the free space it ran through), so openings to the room outside the
        course (seen through gaps in the barriers) are cut off.
Output: <out>/<name>.npz and <name>_rev.npz (loadable by utils/track_gen.to_track, F1Wrapper(tracks=[...]),
        compare_models.py --extra_tracks), <out>/<name>_check.png (map, corridor, centerline, widths, race line).

Steps: drivable cells = free cells connected to the path and within --max_half of it (unknown / walls / beyond =
occupied) -> centerline: the path's first full lap, re-centered between the walls along its normals 3 times (rays
in both directions, smoothed) -> widths w_left / w_right at 0.1 m spacing; pockets / bays (wider than a 1.5 m
rolling median + 0.2 m) excluded, like the generated tracks' smooth corridor (env --bay_cost uses it) -> race line
with the generated tracks' optimizer (min curvature, then min lap time, 0.55 m from the walls).
'''
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, 'src/f1tenth'), os.path.join(REPO, 'src/f1tenth/installation')]


def read_map(d):
    y = {}
    for line in open(f'{d}/map.yaml'):
        if ':' in line:
            k, v = line.split(':', 1)
            y[k.strip()] = v.strip()
    res = float(y['resolution'])
    origin = [float(x) for x in y['origin'].strip('[]').split(',')][:2]
    img = os.path.join(d, y.get('image', 'map.pgm'))
    with open(img, 'rb') as f:
        assert f.readline().strip() == b'P5', 'map.pgm must be binary PGM (P5)'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        W, H = map(int, line.split())
        maxv = int(f.readline())
        g = np.frombuffer(f.read(W * H), np.uint8).reshape(H, W)
    occ_thr = float(y.get('occupied_thresh', 0.65)) * maxv
    free_thr = float(y.get('free_thresh', 0.196)) * maxv
    # ROS: p = (255 - value) / 255 when negate is 0; free = p < free_thresh
    p = (maxv - g.astype(np.float32))
    free = p < free_thr
    # rows: image top = max y -> flip so row 0 = min y (utils/track_gen.rasterize convention)
    return free[::-1].copy(), res, np.array(origin)


def closed_resample(xy, ds):
    from utils.raceline import resample_closed
    out, _ = resample_closed(np.asarray(xy, float), ds)
    return out


def smooth_closed(xy, n):
    k = np.ones(n) / n
    ext = np.concatenate([xy[-n:], xy, xy[:n]])
    return np.column_stack([np.convolve(ext[:, i], k, 'same')[n:-n] for i in range(2)])


def normals(c):
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.column_stack([-t[:, 1], t[:, 0]])              # left


def cast(free, res, org, p, n, max_d):
    '''distance from p along n to the first non-drivable cell'''
    steps = np.arange(0.0, max_d, res * 0.5)
    pts = p[:, None, :] + steps[None, :, None] * n[:, None, :]
    ij = np.floor((pts - org) / res).astype(int)
    H, W = free.shape
    inside = (ij[..., 0] >= 0) & (ij[..., 0] < W) & (ij[..., 1] >= 0) & (ij[..., 1] < H)
    ok = np.zeros(inside.shape, bool)
    ok[inside] = free[ij[..., 1][inside], ij[..., 0][inside]]
    first_bad = np.argmax(~ok, axis=1)
    first_bad[ok.all(axis=1)] = len(steps) - 1
    return steps[first_bad]


def first_lap(path, near=2.0):
    '''the path up to where it first comes back past its start (closest point of the first return within `near`
    m: on later laps the car drives its own line and passes the start 0.5-1 m off the first spot)'''
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    d0 = np.hypot(*(path - path[0]).T)
    # ... driving the same way: a neighbouring straight across one barrier (hairpin, ~2 m away) runs the other way
    # (11/25 review: without this a 45.6 m hairpin course was cut at 32.7 m and closed through the barrier)
    i0 = np.searchsorted(s, s - 0.5).clip(0, len(path) - 1)
    i1 = np.searchsorted(s, s + 0.5).clip(0, len(path) - 1)
    tan = path[i1] - path[i0]
    tan /= np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    same_way = tan @ tan[0] > 0.7
    idx = np.where((s > 15.0) & (d0 < near) & same_way)[0]
    if len(idx) == 0:
        raise SystemExit('[error] the path never returns to its start: drive one full lap (back past the start point)')
    gap = np.where(np.diff(idx) > 1)[0]
    seg = idx[:gap[0] + 1] if len(gap) else idx
    return path[:seg[np.argmin(d0[seg])]]


def build(free_raw, res, org, path, max_half, close_gaps=0.3, blocks=(), ds=0.1):
    # drivable: free cells near the path and connected to it
    H, W = free_raw.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cells = np.column_stack([org[0] + (xx.ravel() + 0.5) * res, org[1] + (yy.ravel() + 0.5) * res])
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(path).query(cells, distance_upper_bound=max_half + res)
    near = np.isfinite(dist).reshape(H, W)
    walls = ~ndimage.binary_opening(free_raw, iterations=1)
    if close_gaps > 0:                                      # gaps in the barriers (LiDAR saw the room through them)
        r = int(round(close_gaps / res))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        walls = ndimage.binary_closing(walls, structure=disk)
    free = ~walls & near
    for x0, y0, x1, y1 in blocks:                            # manual fixes: areas that are not part of the course
        j0, j1 = sorted(np.floor((np.array([x0, x1]) - org[0]) / res).astype(int))
        i0, i1 = sorted(np.floor((np.array([y0, y1]) - org[1]) / res).astype(int))
        free[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1] = False
    lab, _ = ndimage.label(free)
    pij = np.floor((path - org) / res).astype(int)
    seeds = lab[np.clip(pij[:, 1], 0, H - 1), np.clip(pij[:, 0], 0, W - 1)]
    keep = np.bincount(seeds[seeds > 0]).argmax()
    free = lab == keep
    # fill only small holes (noise specks inside the corridor); the island(s) the course goes around stay walls
    holes, nh = ndimage.label(ndimage.binary_fill_holes(free) & ~free)
    if nh:
        size = np.bincount(holes.ravel())
        small = np.isin(holes, np.where(size < 0.5 / res ** 2)[0]) & (holes > 0)
        free = free | small
    # centerline: one lap of the path, re-centered between the walls
    c = closed_resample(smooth_closed(closed_resample(first_lap(path), ds), 15), ds)
    for _ in range(4):
        n = normals(c)
        dl, dr = cast(free, res, org, c, n, max_half), cast(free, res, org, c, -n, max_half)
        c = c + n * ((dl - dr) / 2.0)[:, None]
        c = closed_resample(smooth_closed(c, 15), ds)
    n = normals(c)
    wl, wr = cast(free, res, org, c, n, max_half), cast(free, res, org, c, -n, max_half)
    # smooth corridor: pockets / bays (a sudden widening) do not count as track width
    k = int(1.5 / ds) | 1
    med = lambda w: ndimage.median_filter(w, size=k, mode='wrap')
    wl, wr = np.minimum(wl, med(wl) + 0.2), np.minimum(wr, med(wr) + 0.2)
    occ = np.where(free, 255, 0).astype(np.uint8)
    occ[0, :] = occ[-1, :] = 0
    occ[:, 0] = occ[:, -1] = 0
    return occ, c, wl, wr


def race_line(c, wl, wr):
    from scipy.spatial import cKDTree
    from utils.raceline import min_curvature_line, refine_min_time
    from utils.track_gen import RACELINE_MARGIN, LAP_TIME_LIMITS, OPT_DS, DENSE_DS
    c_opt = closed_resample(c, OPT_DS)
    _, j = cKDTree(c).query(c_opt)
    bl, br = wl[j], wr[j]
    _, off = min_curvature_line(c_opt, bl, br, RACELINE_MARGIN)
    line, _, _ = refine_min_time(c_opt, bl, br, RACELINE_MARGIN, off, **LAP_TIME_LIMITS)
    return closed_resample(line, DENSE_DS)


def save(out, name, mapname, reverse, occ, res, org, c, wl, wr, rl):
    os.makedirs(out, exist_ok=True)
    length = float(np.sum(np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T)))
    np.savez_compressed(f'{out}/{name}.npz', gen_sig=f'real:{mapname}', name=name, map=mapname, reverse=reverse,
                        original=True, occupancy=occ, origin=np.array([org[0], org[1], 0.0]), resolution=res,
                        centerline=c.astype(np.float32), w_left=wl.astype(np.float32), w_right=wr.astype(np.float32),
                        raceline=rl.astype(np.float32), length=length, runoffs=np.zeros((0, 6), np.float32))
    return length


def check_png(path, occ, res, org, c, wl, wr, rl, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, W = occ.shape
    n = normals(c)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})
    ax[0].imshow(occ, origin='lower', cmap='gray', extent=[org[0], org[0] + W * res, org[1], org[1] + H * res])
    ax[0].plot(*c.T, 'b-', lw=0.8, label='centerline')
    ax[0].plot(*(c + n * wl[:, None]).T, 'g-', lw=0.6, label='corridor (w_left / w_right)')
    ax[0].plot(*(c - n * wr[:, None]).T, 'g-', lw=0.6)
    ax[0].plot(*rl.T, 'r-', lw=1.0, label='race line')
    ax[0].plot(*c[0], 'ko', ms=6)
    ax[0].annotate('start', c[0], textcoords='offset points', xytext=(6, 6))
    ax[0].set_aspect('equal'), ax[0].legend(loc='upper right'), ax[0].set_title(title)
    s = np.arange(len(c)) * 0.1
    ax[1].plot(s, wl, label='w_left'), ax[1].plot(s, wr, label='w_right')
    ax[1].set_xlabel('s [m]'), ax[1].set_ylabel('half width [m]'), ax[1].legend(), ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_dir')
    ap.add_argument('--name', required=True, help='track name, e.g. Practice1127 (letters/digits, no _)')
    ap.add_argument('--poses', default=None, help='car path in the map frame (default <map_dir>/poses.csv)')
    ap.add_argument('--max_half', type=float, default=2.5, help='[m] cells farther from the path are walls')
    ap.add_argument('--close_gaps', type=float, default=0.3,
                    help='[m] close gaps in the walls up to twice this wide (barrier joints the LiDAR saw through)')
    ap.add_argument('--block', action='append', default=[], metavar='x0,y0,x1,y1',
                    help='[m, map frame] rectangle that is not part of the course (wall); repeatable. Read the '
                         'coordinates off <name>_check.png (e.g. the room seen through an open barrier)')
    ap.add_argument('--out', default=os.path.join(REPO, 'src/f1tenth/installation/maps_real'))
    a = ap.parse_args()
    assert '_' not in a.name, 'no "_" in --name (utils/track_gen.variant_map splits on it)'
    free, res, org = read_map(a.map_dir)
    P = np.loadtxt(a.poses or f'{a.map_dir}/poses.csv', delimiter=',', skiprows=1)[:, 1:3]
    blocks = [tuple(float(v) for v in b.split(',')) for b in a.block]
    occ, c, wl, wr = build(free, res, org, P, a.max_half, a.close_gaps, blocks)
    for rev in (False, True):
        cc, l, r = (c[::-1].copy(), wr[::-1].copy(), wl[::-1].copy()) if rev else (c, wl, wr)
        rl = race_line(cc, l, r)
        name = a.name + ('_rev' if rev else '')
        L = save(a.out, name, a.name, rev, occ, res, org, cc, l, r, rl)
        check_png(f'{a.out}/{name}_check.png', occ, res, org, cc, l, r, rl, f'{name}: {L:.1f} m, half width {np.r_[l, r].min():.2f}-{np.r_[l, r].max():.2f} m')
        print(f'[track] {a.out}/{name}.npz: length {L:.1f} m, half width left {l.min():.2f}-{l.max():.2f} m, '
              f'right {r.min():.2f}-{r.max():.2f} m, race line {len(rl)} pts')
    print('[next] check the png, then: python3 tools/check_track.py ' + f'{a.out}/{a.name}.npz')


if __name__ == '__main__':
    main()

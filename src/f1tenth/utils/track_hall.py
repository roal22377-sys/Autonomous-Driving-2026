'''
    Narrow "hall" courses like the 2025 competition track (photos in `2025 track/`): double-stacked duct tubes in a
    gym-sized hall, narrow corridors (~1.0-1.6 m; a corridor along the hall border may be wider), a zig-zag infield
    of 90-degree corners and hairpins, neighbouring corridors separated only by ONE tube wall (~0.3-0.45 m), the
    flexible tubes bend into gentle S-curves, small notches where tube ends meet, 50-100 m a lap.

    Layout: a random spanning tree on a coarse grid; the closed walk around the tree (the maze -> Hamiltonian-cycle
    construction on the doubled grid) visits every fine cell once -> a simple closed corridor that fills the hall
    with parallel corridors, 90-degree corners and hairpins (the tree prefers to keep its direction: longer
    straights). A smooth random warp of the plane bends the straights; corners are rounded; widths are the space
    between neighbouring corridors minus a tube wall; no dead-end runoffs (they would cut through the thin walls);
    race-line margin 0.30 m (track_gen: 0.55 m, too much for 1.2 m corridors).

        python -m utils.track_hall --n 30 --seeds 1 2 3 [--out installation/maps_hall]

    Names: Hall<seed>[_rev]_s<len>_w<seed>, seed = block*1000 + k. Held out (eval only): block 3.
    Same file format as utils/track_gen.save_variant (+ 'wall_min', 'pitch').
'''
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

import utils.track_gen as tg
from utils.raceline import resample_closed, min_curvature_line, refine_min_time

HALL_DIR = os.path.join(os.path.dirname(tg.GEN_DIR), 'maps_hall')
HALL_MARGIN = 0.30          # [m] race line to wall: car half width 0.15 + 0.15
WALL = (0.30, 0.45)         # [m] tube wall thickness between neighbouring corridors
PITCH = (1.6, 2.2)          # [m] fine-grid spacing = corridor center to center -> corridors ~1.2-1.9 m; a U-turn
                            # between two neighbouring corridors has a centerline radius of pitch / 2 (car: >= 0.69 m)
HALF_WIDTH_MAX = 1.0        # [m] toward the hall border (outer corridors up to ~2 m)
MIN_HALF = 0.42             # [m] narrowest corridor 0.84 m (car 0.30 m)
HOLDOUT_BLOCK = 3
OUTER_BOOST = 1.05          # [m] half width on the outside of U-turns / tight corners (if the neighbours allow)
TRACK_ERR = 0.15            # [m] race line keeps HALL_MARGIN + this from the walls where the corridor allows
INNER_EXTRA = 0.2           # [m] extra race-line distance from the inner wall in U-turns / tight corners
MIN_LINE_RADIUS = 0.0       # [m] race line radius floor (off: the expert follows tighter lines than the car can, measured by tools/check_short_tracks.py instead)


def is_holdout(path):
    try:
        return int(os.path.basename(path).split('_w')[-1].split('.')[0]) // 1000 == HOLDOUT_BLOCK
    except ValueError:
        return False


def hall_tracks(holdout, hall_dir=HALL_DIR):
    '''(sorted) .npz paths in HALL_DIR: held-out eval courses (holdout=True) or the training ones'''
    return sorted(t for t in glob.glob(os.path.join(hall_dir, '*.npz')) if is_holdout(t) == holdout)


def _spanning_tree(a, b, rng, straight=0.7):
    '''random spanning tree on an a x b grid (randomized DFS that prefers to keep its direction)'''
    seen = np.zeros((a, b), bool)
    edges = []
    start = (int(rng.integers(a)), int(rng.integers(b)))
    stack = [(start, None)]
    seen[start] = True
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while stack:
        (x, y), d = stack[-1]
        nb = [(dx, dy) for dx, dy in dirs if 0 <= x + dx < a and 0 <= y + dy < b and not seen[x + dx, y + dy]]
        if not nb:
            stack.pop()
            continue
        if d in nb and rng.random() < straight:
            dx, dy = d
        else:
            dx, dy = nb[rng.integers(len(nb))]
        n = (x + dx, y + dy)
        seen[n] = True
        edges.append(((x, y), n))
        stack.append((n, (dx, dy)))
    return edges


def _tree_loop(a, b, edges):
    '''the closed walk around the tree on the 2a x 2b fine grid: list of fine cells (each visited once)'''
    conn = {}

    def link(p, q):
        conn.setdefault(p, set()).add(q)
        conn.setdefault(q, set()).add(p)
    tree = set()
    for u, v in edges:
        tree.add((u, v))
        tree.add((v, u))
    for x in range(a):
        for y in range(b):
            fx, fy = 2 * x, 2 * y
            # the 2x2 fine block of a coarse cell: a side is closed (its two fine cells linked) unless a tree edge
            # leaves the cell through it
            if ((x, y), (x, y - 1)) not in tree:
                link((fx, fy), (fx + 1, fy))
            if ((x, y), (x, y + 1)) not in tree:
                link((fx, fy + 1), (fx + 1, fy + 1))
            if ((x, y), (x - 1, y)) not in tree:
                link((fx, fy), (fx, fy + 1))
            if ((x, y), (x + 1, y)) not in tree:
                link((fx + 1, fy), (fx + 1, fy + 1))
    for (x0, y0), (x1, y1) in edges:
        if x0 != x1:                                   # horizontal tree edge: two parallel fine links across it
            xl = max(x0, x1) * 2
            link((xl - 1, 2 * y0), (xl, 2 * y0))
            link((xl - 1, 2 * y0 + 1), (xl, 2 * y0 + 1))
        else:
            yl = max(y0, y1) * 2
            link((2 * x0, yl - 1), (2 * x0, yl))
            link((2 * x0 + 1, yl - 1), (2 * x0 + 1, yl))
    if any(len(v) != 2 for v in conn.values()) or len(conn) != 4 * a * b:
        return None
    start = (0, 0)
    loop, prev, cur = [start], None, start
    while True:
        nxt = sorted(c for c in conn[cur] if c != prev)
        n = nxt[0]
        if n == start:
            break
        loop.append(n)
        prev, cur = cur, n
        if len(loop) > 4 * a * b:
            return None
    return loop if len(loop) == 4 * a * b else None


def _fillet(verts, rmax, ds):
    '''closed polyline -> straight pieces + circular arcs at the vertices; each arc's tangent length is at most half
       of each neighbouring segment (90 deg between two vertices a pitch apart: radius pitch / 2)'''
    V = np.asarray(verts, float)
    n = len(V)
    seg = np.linalg.norm(np.roll(V, -1, 0) - V, axis=1)                # seg[i]: V[i] -> V[i+1]
    pts = []
    for i in range(n):
        p, c, q = V[i - 1], V[i], V[(i + 1) % n]
        u, w = (c - p) / np.linalg.norm(c - p), (q - c) / np.linalg.norm(q - c)
        th = np.arccos(np.clip(u @ w, -1.0, 1.0))                     # turning angle
        if th < 1e-3:
            pts.append(c[None])
            continue
        t = min(0.5 * seg[i - 1], 0.5 * seg[i], rmax * np.tan(th / 2))  # tangent length
        r = t / np.tan(th / 2)
        a, b = c - u * t, c + w * t
        side = np.sign(u[0] * w[1] - u[1] * w[0])                       # +1: left turn
        nrm = np.array([-u[1], u[0]]) * side
        o = a + nrm * r                                                 # arc center
        a0 = np.arctan2(a[1] - o[1], a[0] - o[0])
        m = max(int(np.ceil(r * th / ds)), 2)
        ang = a0 + side * th * np.arange(m + 1) / m
        pts.append(np.column_stack([o[0] + r * np.cos(ang), o[1] + r * np.sin(ang)]))
    # straight pieces between the arcs come from the resampling (points are in order along the loop)
    return _polyline(np.vstack(pts), ds)


def _corners(V):
    '''drop the vertices of a closed polyline where it goes straight on'''
    u = np.roll(V, -1, 0) - V
    w = V - np.roll(V, 1, 0)
    turn = np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]) > 1e-9
    return V[turn]


def _warp(pts, rng, amp=(0.10, 0.25), wavelength=(6.0, 12.0), n=3):
    '''smooth random displacement of the plane (the flexible tubes bend: straights become gentle S-curves).
       Small amplitude x wavenumber keeps it a smooth bijection: corridors never cross and keep ~their spacing.'''
    out = pts.copy()
    for ax in (0, 1):
        for _ in range(n):
            lam, A = rng.uniform(*wavelength), rng.uniform(*amp) / np.sqrt(n)
            th = rng.uniform(0, 2 * np.pi)
            k = 2 * np.pi / lam * np.array([np.cos(th), np.sin(th)])
            out[:, ax] += A * np.sin(pts @ k + rng.uniform(0, 2 * np.pi))
    return out


def _polyline(corners, ds):
    '''closed polygon -> points every ~ds along its straight edges (a spline through sparse corners would round
       the whole course)'''
    out = []
    for p, q in zip(corners, np.roll(corners, -1, 0)):
        if np.linalg.norm(q - p) < 1e-6:                                 # repeated point (arcs that meet)
            continue
        n = max(int(np.ceil(np.linalg.norm(q - p) / ds)), 1)
        out.append(p + (q - p) * np.arange(n)[:, None] / n)
    return np.vstack(out)


def hall_centerline(seed, target=(50.0, 100.0), tries=300):
    '''(centerline (N, 2) at tg.DENSE_DS, pitch) of a random hall course, or (None, None)'''
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        pitch = rng.uniform(*PITCH)
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))     # hall ~ (2a x 2b) pitch, aspect <= 2
        if max(a, b) > 2 * min(a, b):
            continue
        if not target[0] <= 4 * a * b * pitch <= target[1] * 1.1:
            continue
        loop = _tree_loop(a, b, _spanning_tree(a, b, rng, straight=rng.uniform(0.5, 0.9)))
        if loop is None:
            continue
        # bend the tubes (warp the grid points: straights become gentle polylines), then round every corner with
        # a circular arc as large as the neighbouring segments allow (U-turn: a semicircle of radius ~pitch / 2)
        dense = _fillet(_corners(np.array(loop, float) * pitch), rng.uniform(0.8, 1.3), 0.05)
        center, length = resample_closed(_warp(dense, rng), tg.DENSE_DS)
        if target[0] <= length <= target[1]:
            return center, pitch
    return None, None


def _tangents(center):
    t = np.roll(center, -1, 0) - np.roll(center, 1, 0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _side_clearance(center, s, length, min_ds):
    '''distance to the nearest OTHER part of the course on the left / right of each point (20 m: none)'''
    t = _tangents(center)
    cl, cr = np.full(len(center), 20.0), np.full(len(center), 20.0)
    for i, nb in enumerate(cKDTree(center).query_ball_point(center, r=6.0)):
        nb = np.asarray(nb, int)
        ds = np.abs(s[nb] - s[i])
        nb = nb[np.minimum(ds, length - ds) > min_ds]
        if not len(nb):
            continue
        v = center[nb] - center[i]
        d = np.linalg.norm(v, axis=1)
        left = t[i, 0] * v[:, 1] - t[i, 1] * v[:, 0] > 0
        if left.any():
            cl[i] = d[left].min()
        if (~left).any():
            cr[i] = d[~left].min()
    return cl, cr


def wall_gaps(center, wl, wr, min_ds=3.0):
    '''thinnest wall [m] between two different parts of the course (both corridors' free space subtracted)'''
    N = len(center)
    length = float(np.sum(np.linalg.norm(np.roll(center, -1, 0) - center, axis=1)))
    s = np.arange(N) * length / N
    t = _tangents(center)
    best = 20.0
    for i, nb in enumerate(cKDTree(center).query_ball_point(center, r=4.0)):
        nb = np.asarray(nb, int)
        ds = np.abs(s[nb] - s[i])
        nb = nb[np.minimum(ds, length - ds) > min_ds]
        if not len(nb):
            continue
        v = center[nb] - center[i]
        d = np.linalg.norm(v, axis=1)
        si = np.where(t[i, 0] * v[:, 1] - t[i, 1] * v[:, 0] > 0, wl[i], wr[i])
        sj = np.where(t[nb, 0] * -v[:, 1] - t[nb, 1] * -v[:, 0] > 0, wl[nb], wr[nb])
        best = min(best, float(np.min(d - si - sj)))
    return best


def _curvature(xy):
    '''signed curvature of a closed line (points ~evenly spaced), from neighbours 3 samples apart'''
    a, b, c = np.roll(xy, 3, 0), xy, np.roll(xy, -3, 0)
    ab, bc, ca = b - a, c - b, a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    return 2 * cross / np.maximum(den, 1e-9)


def min_radius(xy):
    '''smallest turning radius of a closed line [m] (curvature over +-0.3 m, 5-sample median against noise)'''
    k = np.abs(_curvature(xy))
    k = np.median(np.stack([np.roll(k, i) for i in range(-2, 3)]), axis=0)
    return float(1.0 / max(k.max(), 1e-6))


def _circ_smooth(x, n):
    ker = np.ones(2 * n + 1) / (2 * n + 1)
    return np.convolve(np.r_[x[-n:], x, x[:n]], ker, 'valid')


def make_hall(seed, reverse=False, target=(50.0, 100.0)):
    center, pitch = hall_centerline(seed, target)
    if center is None:
        return None
    return hall_course(center, pitch, np.random.default_rng(seed + 7919), f'Hall{seed}', seed, reverse)


# 2025 competition course, approximated from the photos in `2025 track/` (no measured map: corridor positions
# +-0.5 m). Hall ~10 x 16 m: outer straight ~1.8 m wide along one long side, a U along the far end, a tight
# two-hairpin serpentine (corridor pitch 1.6 m), an S-shaped middle corridor, a 90-degree jog and a wide hairpin
# back to the start; ~60 m. Corner points [m], counter-clockwise.
COURSE_2025 = [(9.0, 1.2), (9.0, 15.0), (1.0, 15.0), (1.0, 10.0), (2.6, 10.0), (2.6, 13.4), (4.2, 13.4),
               (4.2, 7.5), (6.6, 7.5), (6.6, 4.2), (1.0, 4.2), (1.0, 1.2)]


def make_2025(reverse=False, seed=2025):
    rng = np.random.default_rng(seed)
    dense = _fillet(np.array(COURSE_2025, float), 1.2, 0.05)
    center, _ = resample_closed(_warp(dense, rng, amp=(0.15, 0.25)), tg.DENSE_DS)
    return hall_course(center, 1.6, rng, 'Course2025', seed, reverse, wall=0.35, free=0.9, fill=0.97)


def hall_course(center, pitch, rng, name, seed, reverse=False, wall=None, free=None, fill=None):
    if reverse:
        center = center[::-1].copy()
    center, length = resample_closed(center, tg.DENSE_DS)
    N = len(center)
    s = np.arange(N) * (length / N)
    wall = rng.uniform(*WALL) if wall is None else wall
    # half width: the space to the neighbouring corridor minus one tube wall, split evenly; toward the hall border
    # (no neighbour) up to HALF_WIDTH_MAX. min_ds > a hairpin's own arc so both legs of a hairpin count as neighbours.
    cl, cr = _side_clearance(center, s, length, min_ds=1.2 * pitch + 1.0)
    capl = _circ_smooth(np.maximum((cl - wall) / 2.0, MIN_HALF), 3)
    capr = _circ_smooth(np.maximum((cr - wall) / 2.0, MIN_HALF), 3)
    capl, capr = np.minimum(capl, (cl - wall) / 2.0).clip(MIN_HALF), np.minimum(capr, (cr - wall) / 2.0).clip(MIN_HALF)
    free = rng.uniform(0.6, HALF_WIDTH_MAX) if free is None else free
    fill = rng.uniform(0.85, 1.0) if fill is None else fill             # how much of the space the corridor takes
    var = tg._smooth_noise(s, length, rng, corr_len=6.0, std=0.05)
    asym = tg._smooth_noise(s, length, rng, corr_len=8.0, std=0.03)
    wl = _circ_smooth(np.minimum(free, capl * fill) + var + asym, 10)
    wr = _circ_smooth(np.minimum(free, capr * fill) + var - asym, 10)
    # U-turns / tight corners: the tube on the outside is a wide arc (a tube cannot bend tight; photos: lots of room
    # around the tube ends) -> the outside of curves under 1.5 m radius gets up to OUTER_BOOST of width
    kap = _curvature(center)                                            # > 0: turning left
    tight = _circ_smooth((np.abs(kap) > 1 / 1.5).astype(float), 12) > 0.05
    wl = np.where(tight & (kap < 0), np.maximum(wl, OUTER_BOOST), wl)  # right turn: outside is the left
    wr = np.where(tight & (kap > 0), np.maximum(wr, OUTER_BOOST), wr)
    wl, wr = _circ_smooth(wl, 5), _circ_smooth(wr, 5)
    wl, wr = np.clip(wl, MIN_HALF, np.maximum(capl, MIN_HALF)), np.clip(wr, MIN_HALF, np.maximum(capr, MIN_HALF))
    # tube surface waviness; small notches where tube ends meet (a few cm to 2 dm deep; no runoffs)
    wl_occ = np.clip(wl + tg._smooth_noise(s, length, rng, 0.8, 0.02), MIN_HALF - 0.03, capl + 0.03)
    wr_occ = np.clip(wr + tg._smooth_noise(s, length, rng, 0.8, 0.02), MIN_HALF - 0.03, capr + 0.03)
    for _ in range(rng.poisson(length / 20.0)):
        c, half, depth = rng.uniform(0, length), rng.uniform(0.1, 0.3), rng.uniform(0.05, 0.2)
        dist = np.abs(s - c)
        bump = depth * (np.minimum(dist, length - dist) < half)
        if rng.random() < 0.5:
            wl_occ = np.maximum(wl_occ, np.minimum(wl + bump, capl + 0.03))
        else:
            wr_occ = np.maximum(wr_occ, np.minimum(wr + bump, capr + 0.03))
    gap = wall_gaps(center, wl_occ, wr_occ)
    if gap < 0.2:                                                       # corridors would (nearly) touch
        return None
    c_opt, _ = resample_closed(center, tg.OPT_DS)
    _, j = cKDTree(center).query(c_opt)
    bl, br = np.minimum(wl, wl_occ)[j], np.minimum(wr, wr_occ)[j]
    # the optimizer checks the walls only every OPT_DS (0.5 m) across the centerline: in a U-turn the line cut the
    # inner tube end between two checks and the expert clipped it (11/26) -> keep INNER_EXTRA more on the inside there
    # + the expert's tracking error (mean 0.15 m, p95 0.28 m): with 0.30 m alone it touched the walls of 1.2-1.5 m
    # corridors on straights (11/26); in a narrow corridor the line then simply stays near the middle
    inner = INNER_EXTRA * tight[j]
    bl = np.maximum(bl - TRACK_ERR - np.where(kap[j] > 0, inner, 0.0), HALL_MARGIN + 0.02)
    br = np.maximum(br - TRACK_ERR - np.where(kap[j] < 0, inner, 0.0), HALL_MARGIN + 0.02)
    _, off = min_curvature_line(c_opt, bl, br, HALL_MARGIN)
    line_opt, _, _ = refine_min_time(c_opt, bl, br, HALL_MARGIN, off, **tg.LAP_TIME_LIMITS)
    raceline, _ = resample_closed(line_opt, tg.DENSE_DS)
    r_line = min_radius(raceline)
    if r_line < MIN_LINE_RADIUS:                                        # the car cannot turn that tight
        return None
    occ, origin = tg.rasterize(center, wl_occ, wr_occ, ())
    return {
        'name': f"{name}{'_rev' if reverse else ''}_s{int(round(length))}_w{seed}",
        'map': name, 'reverse': reverse, 'original': False,
        'occupancy': occ, 'origin': np.array([origin[0], origin[1], 0.0]), 'resolution': tg.RESOLUTION,
        'centerline': center.astype(np.float32), 'w_left': wl.astype(np.float32), 'w_right': wr.astype(np.float32),
        'raceline': raceline.astype(np.float32), 'length': length, 'runoffs': np.zeros((0, 6), np.float32),
        'wall_min': gap, 'pitch': pitch, 'line_rmin': r_line,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=20, help='courses per seed block')
    ap.add_argument('--seeds', type=int, nargs='*', default=[1], help=f'seed blocks (seed = block*1000 + k); '
                    f'block {HOLDOUT_BLOCK} = held-out eval courses')
    ap.add_argument('--out', default=HALL_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for rev in (False, True):                       # the 2025 course copy: always an eval (held-out) course
        v = make_2025(rev)
        v['name'] = v['name'].replace('_w2025', f'_w{HOLDOUT_BLOCK}025')
        tg.save_variant(v, a.out)
        print(f"[track] {v['name']}: {float(v['length']):.1f} m, thinnest wall {v['wall_min']:.2f} m")
    for blk in a.seeds:
        for k in range(a.n):
            seed = blk * 1000 + k
            for rev in (False, True):
                v = make_hall(seed, rev)
                if v is None:
                    print(f'[skip] {seed}: no valid layout')
                    break
                tg.save_variant(v, a.out)
                w = v['w_left'] + v['w_right']
                print(f"[track] {v['name']}: {float(v['length']):.1f} m, pitch {v['pitch']:.2f}, width "
                      f"{w.min():.2f}-{np.median(w):.2f}-{w.max():.2f} m, thinnest wall {v['wall_min']:.2f} m, "
                      f"line radius >= {v['line_rmin']:.2f} m", flush=True)


if __name__ == '__main__':
    main()

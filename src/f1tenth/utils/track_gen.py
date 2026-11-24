'''
    Randomized training tracks built from the F1 map centerlines.

    Why: every original map is exactly 2.2 m wide with continuous walls, so the policy only ever
    saw side walls at ~1.1 m. The real track is 2.4 m wide on median (4.1 m p90) and has open
    side areas, and the policy crashed whenever a side reading left the training range.

    A variant = one map centerline with
      - a random, smoothly varying width (left/right independent)          -> width diversity
      - a few "pockets": short side areas much wider than the track        -> open-side readings
        (smooth bumps or sharp rectangular alcoves)
      - corner run-offs: a dead-end bay that continues straight ahead at a corner entry, on the
        outside of the turn (11/23 real car: it drove straight into such wide closed areas instead
        of turning)                                                       -> "open but closed" readings
      - slightly rough walls (real walls read wavy by a few cm)
      - optionally the reverse driving direction                           -> both directions
      - its own minimum-curvature race line (utils/raceline.py)            -> expert follows it
    Walls are solid (everything outside the corridor is occupied). Widths are capped so the
    corridor never merges with a neighbouring part of the same track (no shortcuts).

    Variants are generated once (python -m utils.track_gen ...) into installation/maps_gen/
    and loaded per episode by utils/env.py.

    Every file stores a signature of the generator (its code + settings, see generator_signature()).
    Re-running the generator regenerates the files whose signature differs (made by an older
    generator) and skips the up-to-date ones, so an interrupted run resumes and a changed
    generator never leaves old tracks behind. The training scripts refuse to start on stale tracks.
'''

import os
import glob
import hashlib
import inspect
import numpy as np
from scipy.spatial import cKDTree

from utils.raceline import resample_closed, tangents_normals, curvature, min_curvature_line, refine_min_time

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(HERE, '..', 'installation', 'maps')
# F1_TRACK_DIR overrides it (experiments with another track set; workers inherit the variable)
GEN_DIR = os.environ.get('F1_TRACK_DIR', os.path.join(HERE, '..', 'installation', 'maps_gen'))

RESOLUTION = 0.08          # [m/cell], originals use ~0.085
CAR_HALF_WIDTH = 0.15         # real car is 0.30 m wide, 0.60 m long
RACELINE_MARGIN = 0.55     # race line keeps this far from the walls: half width 0.15 + 0.40 for tracking error under ~0.2 s steering delay
DENSE_DS = 0.1             # centerline / race line waypoint spacing used by the env and expert
OPT_DS = 0.5               # spacing for the race line optimization
# car limits for the lap-time refinement (same as the expert: train_il_s2r.EXPERT, a_max nominal)
LAP_TIME_LIMITS = dict(v_min=0.5, v_max=8.0, a_lat=3.5, a_acc=3.0, a_brake=3.0)

# dead ends (randomized variants only)
RECT_POCKET_PROB = 0.5     # share of pockets that are sharp rectangular alcoves instead of smooth bumps
RUNOFF_PROB = 0.5          # chance that a corner gets a straight-ahead dead-end bay at its entry
RUNOFF_MIN_CURV = 1 / 5.0  # [1/m] corners tighter than radius 5 m
RUNOFF_LEN = (2.0, 7.0)    # [m] bay length from the corner entry (11/23: was 2-5; a 4-7 m bay
                           #     made IL run 8 drive in on 13 of 27 tracks it otherwise finished)
RUNOFF_EXTRA_W = (0.0, 2.5)  # [m] bay wider than the track on the outside (was 0-1): the wider, the more it looks like the road
WALL_ROUGH_STD = 0.05      # [m] wall waviness
WALL_ROUGH_CORR = 0.6      # [m] its correlation length


def load_centerline(map_name):
    a = np.loadtxt(os.path.join(MAP_DIR, map_name, f'{map_name}_centerline.csv'), delimiter=',', comments='#')
    xy = a[:, :2]
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-3:
        xy = xy[:-1]
    return xy


def _smooth_noise(s, length, rng, corr_len, std):
    '''Smooth periodic random function of arclength s (random Fourier series).'''
    kmax = max(1, int(length / corr_len))
    out = np.zeros_like(s)
    for k in range(1, kmax + 1):
        amp = rng.normal(0.0, 1.0) / np.sqrt(kmax)
        out += amp * np.sin(2 * np.pi * k * s / length + rng.uniform(0, 2 * np.pi))
    return std * out / (np.std(out) + 1e-9)


def _self_clearance(xy, s, length, min_ds=6.0):
    '''Distance from each point to the nearest point of a *different* part of the track.'''
    tree = cKDTree(xy)
    reach = 2 * 2.2 + 2 * 3.5 + 1.0          # widest corridor incl. pockets on both sides
    out = np.full(len(xy), 10.0)
    for i, nb in enumerate(tree.query_ball_point(xy, r=reach)):
        nb = np.asarray(nb, dtype=int)
        ds = np.abs(s[nb] - s[i])
        ds = np.minimum(ds, length - ds)
        nb = nb[ds > min_ds]
        if len(nb):
            out[i] = np.min(np.linalg.norm(xy[nb] - xy[i], axis=1))
    return out


def make_variant(map_name, seed, reverse=False, original=False):
    '''
        original=True keeps the original geometry (1.1 m each side, no pockets).
        returns a dict of arrays (see save/load below).
    '''
    rng = np.random.default_rng(seed)
    center, length = resample_closed(load_centerline(map_name), DENSE_DS)
    if reverse:
        center = center[::-1].copy()
    N = len(center)
    s = np.arange(N) * (length / N)

    if original:
        wl = np.full(N, 1.1)
        wr = np.full(N, 1.1)
        pocket_l = np.zeros(N)
        pocket_r = np.zeros(N)
    else:
        base = rng.uniform(0.85, 1.6)                                  # half width -> 1.7..3.2 m track
        var = _smooth_noise(s, length, rng, corr_len=15.0, std=0.18)    # slow width changes
        asym = _smooth_noise(s, length, rng, corr_len=20.0, std=0.12)   # centerline not centered
        wl = np.clip(base + var + asym, 0.7, 2.2)
        wr = np.clip(base + var - asym, 0.7, 2.2)
        # pockets: open side areas (doorways, gaps, wider sections next to the barrier)
        pocket_l, pocket_r = np.zeros(N), np.zeros(N)
        for _ in range(rng.poisson(length / 45.0)):
            c = rng.uniform(0, length)
            half = rng.uniform(0.5, 2.0)
            depth = rng.uniform(1.0, 3.5)
            dist = np.abs(s - c)
            dist = np.minimum(dist, length - dist)
            if rng.random() < RECT_POCKET_PROB:
                bump = depth * (dist < half)                                   # sharp alcove
            else:
                bump = depth * np.clip(1.0 - dist / half, 0.0, 1.0) ** 0.5     # smooth bump
            if rng.random() < 0.5:
                pocket_l = np.maximum(pocket_l, bump)
            else:
                pocket_r = np.maximum(pocket_r, bump)

    # never merge with a neighbouring part of the track
    clear = _self_clearance(center, s, length)
    cap = np.maximum(clear / 2.0 - 0.15, 0.6)
    wl, wr = np.minimum(wl, cap), np.minimum(wr, cap)
    wl_occ = np.minimum(wl + pocket_l, cap)
    wr_occ = np.minimum(wr + pocket_r, cap)
    runoffs = []
    if not original:
        # rough walls: only ever widen the occupied boundary inwards a little or outwards a little,
        # never into the race line margin (0.55 m) -> keep at least 0.45 m of half width
        wl_occ = np.maximum(wl_occ + _smooth_noise(s, length, rng, WALL_ROUGH_CORR, WALL_ROUGH_STD), 0.45)
        wr_occ = np.maximum(wr_occ + _smooth_noise(s, length, rng, WALL_ROUGH_CORR, WALL_ROUGH_STD), 0.45)
        runoffs = _corner_runoffs(center, s, length, wl, wr, wl_occ, wr_occ, rng)

    # race line on the drivable corridor (pockets excluded): minimum curvature, then refined
    # for minimum lap time with the car's limits (late apexes etc., utils/raceline.py)
    c_opt, _ = resample_closed(center, OPT_DS)
    tree = cKDTree(center)
    _, j = tree.query(c_opt)
    # bounded by the rough wall where it comes closer than the smooth width (11/24: the roughness was added after
    # the line, leaving 0.33 m from the car's center to the wall instead of 0.55; pockets only widen, ignored here)
    bl, br = np.minimum(wl, wl_occ)[j], np.minimum(wr, wr_occ)[j]
    _, off = min_curvature_line(c_opt, bl, br, RACELINE_MARGIN)
    line_opt, _, _ = refine_min_time(c_opt, bl, br, RACELINE_MARGIN, off, **LAP_TIME_LIMITS)
    raceline, _ = resample_closed(line_opt, DENSE_DS)

    occ, origin = rasterize(center, wl_occ, wr_occ, runoffs)
    return {
        'name': f"{map_name}{'_rev' if reverse else ''}{'' if original else f'_w{seed}'}",
        'map': map_name, 'reverse': reverse, 'original': original,
        'occupancy': occ, 'origin': np.array([origin[0], origin[1], 0.0]), 'resolution': RESOLUTION,
        'centerline': center.astype(np.float32), 'w_left': wl.astype(np.float32), 'w_right': wr.astype(np.float32),
        'raceline': raceline.astype(np.float32), 'length': length,
        'runoffs': np.array(runoffs, dtype=np.float32).reshape(-1, 6),
    }


def _corner_runoffs(center, s, length, wl, wr, wl_occ, wr_occ, rng):
    '''
        Dead-end bays at corner entries: a rectangle that starts at the corridor where the corner
        begins and continues straight along the incoming direction, covering the track width plus
        some extra on the outside of the turn. Returns rows [x0, y0, yaw, length, lat_lo, lat_hi]:
        entry point, its heading, bay length and lateral extent (+ = left) in that frame.
        Skipped if it would come near another part of the track (no shortcuts).
    '''
    N = len(center)
    t, n = tangents_normals(center)
    kappa = np.convolve(np.r_[curvature(center)[-15:], curvature(center), curvature(center)[:15]],
                        np.ones(31) / 31, mode='valid')                  # +-1.5 m smoothing
    ds = length / N
    out = []
    # corners = local maxima of |curvature| above the threshold, at least 8 m apart
    cand = [i for i in range(N) if abs(kappa[i]) > RUNOFF_MIN_CURV
            and abs(kappa[i]) >= abs(kappa[(i - 1) % N]) and abs(kappa[i]) >= abs(kappa[(i + 1) % N])]
    taken = []
    tree = cKDTree(center)
    reach = max(float(np.max(wl_occ)), float(np.max(wr_occ)))
    for apex in cand:
        if any(min(abs(apex - j), N - abs(apex - j)) * ds < 8.0 for j in taken):
            continue
        taken.append(apex)
        if rng.random() > RUNOFF_PROB:
            continue
        # corner entry: walk back from the apex to where the curvature drops below 35 % of the apex value
        i_e = apex
        for k in range(1, int(6.0 / ds)):
            j = (apex - k) % N
            if abs(kappa[j]) < 0.35 * abs(kappa[apex]):
                i_e = j
                break
        left_turn = kappa[apex] > 0
        L = rng.uniform(*RUNOFF_LEN)
        extra = rng.uniform(*RUNOFF_EXTRA_W)
        lo, hi = -float(wr[i_e]), float(wl[i_e])
        if left_turn:
            lo -= extra             # outside of a left turn is the right side
        else:
            hi += extra
        yaw = float(np.arctan2(t[i_e, 1], t[i_e, 0]))
        x0, y0 = center[i_e]
        # the bay must stay clear of every part of the track more than ~L + 6 m away along it
        u = np.linspace(0, L + 1.0, 12)
        v = np.linspace(lo, hi, 6)
        U, V = np.meshgrid(u, v)
        P = np.stack([x0 + U * np.cos(yaw) - V * np.sin(yaw), y0 + U * np.sin(yaw) + V * np.cos(yaw)], -1).reshape(-1, 2)
        ok = True
        for nb in tree.query_ball_point(P, r=reach + 0.8):
            nb = np.asarray(nb, dtype=int)
            if not len(nb):
                continue
            dsi = np.abs(s[nb] - s[i_e])
            dsi = np.minimum(dsi, length - dsi)
            if np.any(dsi > L + 6.0):
                ok = False
                break
        if ok:
            out.append([x0, y0, yaw, L, lo, hi])
    return out


def rasterize(center, wl, wr, runoffs=()):
    '''
        Occupancy grid (255 free / 0 occupied), rows = y, cols = x, like the original map images.
        runoffs: extra free rectangles [x0, y0, yaw, length, lat_lo, lat_hi] (see _corner_runoffs).
    '''
    pad = float(max(wl.max(), wr.max())) + 1.0 + max([r[3] for r in runoffs], default=0.0)
    xmin, ymin = center.min(axis=0) - pad
    xmax, ymax = center.max(axis=0) + pad
    W = int(np.ceil((xmax - xmin) / RESOLUTION))
    H = int(np.ceil((ymax - ymin) / RESOLUTION))
    xs = xmin + (np.arange(W) + 0.5) * RESOLUTION
    ys = ymin + (np.arange(H) + 0.5) * RESOLUTION
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    t, n = tangents_normals(center)
    tree = cKDTree(center)
    d, i = tree.query(pts, distance_upper_bound=pad, workers=1)
    occ = np.zeros(len(pts), dtype=np.uint8)
    ok = np.isfinite(d)
    ii = i[ok]
    rel = pts[ok] - center[ii]
    lat = np.sum(rel * n[ii], axis=1)
    free = (lat <= wl[ii]) & (lat >= -wr[ii])
    idx = np.where(ok)[0][free]
    occ[idx] = 255
    for x0, y0, yaw, L, lo, hi in runoffs:
        rel = pts - np.array([x0, y0])
        u = rel[:, 0] * np.cos(yaw) + rel[:, 1] * np.sin(yaw)
        v = -rel[:, 0] * np.sin(yaw) + rel[:, 1] * np.cos(yaw)
        occ[(u >= 0) & (u <= L) & (v >= lo) & (v <= hi)] = 255
    occ = occ.reshape(H, W)
    occ[0, :] = occ[-1, :] = 0
    occ[:, 0] = occ[:, -1] = 0
    return occ, (xmin, ymin)


GEN_SETTINGS = ('RESOLUTION', 'CAR_HALF_WIDTH', 'RACELINE_MARGIN', 'DENSE_DS', 'OPT_DS', 'LAP_TIME_LIMITS',
                'RECT_POCKET_PROB', 'RUNOFF_PROB', 'RUNOFF_MIN_CURV', 'RUNOFF_LEN', 'RUNOFF_EXTRA_W',
                'WALL_ROUGH_STD', 'WALL_ROUGH_CORR')
_SIG = None


def generator_signature():
    '''
        short hash of everything that shapes a variant: the generator functions' code, the race line
        code (utils/raceline.py) and the settings above. Comments elsewhere in this file and the
        __main__ block do not count, so editing them does not force a regeneration.
    '''
    global _SIG
    if _SIG is None:
        from utils import raceline
        h = hashlib.sha1()
        for fn in (load_centerline, _smooth_noise, _self_clearance, make_variant, _corner_runoffs, rasterize):
            h.update(inspect.getsource(fn).encode())
        h.update(inspect.getsource(raceline).encode())
        h.update(repr([(k, globals()[k]) for k in GEN_SETTINGS]).encode())
        _SIG = h.hexdigest()[:12]
    return _SIG


def file_signature(path):
    '''generator signature stored in a variant file ('' for files from before signatures, or unreadable)'''
    try:
        with np.load(path) as z:
            return str(z['gen_sig']) if 'gen_sig' in z.files else ''
    except Exception:
        return ''


def stale_variants(files=None):
    '''variant files not made by the current generator (None = all files in GEN_DIR)'''
    files = list_variants() if files is None else files
    sig = generator_signature()
    return [f for f in files if file_signature(f) != sig]


def require_fresh_variants(files):
    '''raise if any of the files was made by an older generator (training on old geometry by mistake)'''
    if not files:
        raise FileNotFoundError(f'no generated tracks in {GEN_DIR}: run `python -m utils.track_gen 6 22` in src/f1tenth')
    old = stale_variants(files)
    if old and os.environ.get('F1_ALLOW_STALE_TRACKS') == '1':      # deliberate experiments only
        print(f'[warning] {len(old)} of {len(files)} tracks are from another generator (F1_ALLOW_STALE_TRACKS=1)')
    elif old:
        raise RuntimeError(f'{len(old)} of {len(files)} generated tracks were made by an older track generator '
                           f'(e.g. {os.path.basename(old[0])}). Regenerate them first: '
                           f'`python -m utils.track_gen 6 22` in src/f1tenth (only the stale ones are rebuilt).')


def save_variant(v, out_dir=GEN_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{v['name']}.npz")
    tmp = path[:-4] + '.tmp.npz'        # write then rename: an interrupted save never leaves a half file
    np.savez_compressed(tmp, gen_sig=generator_signature(), **{k: np.asarray(val) for k, val in v.items()})
    os.replace(tmp, path)
    return path


def to_track(path):
    '''Load a saved variant as a gym Track (with centerline & raceline Raceline objects).'''
    from f1tenth_gym.envs.track import Track, Raceline
    from f1tenth_gym.envs.track.track import TrackSpec
    z = np.load(path)
    occ = z['occupancy'].astype(np.float32)
    spec = TrackSpec(name=str(z['name']), image=None, resolution=float(z['resolution']),
                     origin=tuple(float(x) for x in z['origin']), negate=0,
                     occupied_thresh=0.45, free_thresh=0.196, width=0.0)

    def line(xy):
        xs, ys = xy[:, 0].astype(np.float32), xy[:, 1].astype(np.float32)
        return Raceline(xs=xs, ys=ys, velxs=np.full_like(xs, 8.0))

    track = Track(spec=spec, occupancy_map=occ, centerline=line(z['centerline']),
                  raceline=line(z['raceline']), name=str(z['name']))
    track.w_left, track.w_right = z['w_left'], z['w_right']
    track.gen_length = float(z['length'])
    return track


def list_variants(maps=None, gen_dir=GEN_DIR):
    files = sorted(f for f in glob.glob(os.path.join(gen_dir, '*.npz')) if not f.endswith('.tmp.npz'))
    if maps is not None:
        files = [f for f in files if variant_map(f) in maps]
    return files


def variant_map(path):
    '''map name of a variant file: Sochi.npz, Sochi_rev.npz, Sochi_w101.npz -> Sochi'''
    return os.path.basename(path)[:-4].split('_')[0]


def _job(job):
    map_name, seed, reverse, original = job
    name = f"{map_name}{'_rev' if reverse else ''}{'' if original else f'_w{seed}'}"
    path = os.path.join(GEN_DIR, f'{name}.npz')
    # skip only files the current generator made: resumes an interrupted run, rebuilds old geometry
    if os.path.exists(path) and file_signature(path) == generator_signature():
        return path, 'kept'
    old = os.path.exists(path)
    return save_variant(make_variant(map_name, seed, reverse, original)), ('rebuilt' if old else 'new')


if __name__ == '__main__':
    # python -m utils.track_gen [variants_per_map=16] [workers]  generate / bring up to date
    # python -m utils.track_gen --check                          only report stale files
    import sys
    import multiprocessing as mp
    sys.path.insert(0, os.path.join(HERE, '..', 'installation'))
    from main_il import MAPS
    if '--check' in sys.argv:
        files = list_variants()
        old = stale_variants(files)
        print(f'generator {generator_signature()}: {len(files) - len(old)} up to date, {len(old)} stale'
              + (f' (e.g. {os.path.basename(old[0])})' if old else ''))
        sys.exit(1 if old or not files else 0)
    per_map = int(sys.argv[1]) if len(sys.argv) > 1 else 16    # 11/24: 6 -> 16 (PPO memorized the 5 training variants per layout)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    jobs = []
    for m in MAPS:
        for rev in (False, True):
            jobs.append((m, 0, rev, True))                              # original geometry
            jobs += [(m, 100 + k, rev, False) for k in range(per_map)]  # randomized
    print(f'generator signature {generator_signature()}', flush=True)
    count = {'kept': 0, 'rebuilt': 0, 'new': 0}
    with mp.get_context('fork').Pool(workers) as p:
        for i, (path, what) in enumerate(p.imap_unordered(_job, jobs)):
            count[what] += 1
            print(f'[{i + 1}/{len(jobs)}] {os.path.basename(path)} ({what})', flush=True)
    for f in glob.glob(os.path.join(GEN_DIR, '*.tmp.npz')):      # leftovers of an interrupted save
        os.remove(f)
    print(f'done: {count["new"]} new, {count["rebuilt"]} rebuilt (made by an older generator), '
          f'{count["kept"]} already up to date')

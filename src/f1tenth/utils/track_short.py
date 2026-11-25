'''
    Short tracks (~70-110 m a lap): the F1 centerlines of utils/track_gen.py scaled down, built by the same
    make_variant (random widths, pockets, corner dead-ends, rough walls, race line). The generated training tracks
    are 260-554 m a lap, the F1TENTH competition course is expected <= 100 m (11/25): shorter straights, corners
    closer together. Maps whose scaled centerline gets tighter than MIN_RADIUS are skipped (or scaled less).

        python -m utils.track_short [--maps ...] [--seeds 301 302] [--target 90] [--out installation/maps_short]

    IL 13 / PPO set (11/25): `--random 8 --seeds 1 2 3 4 5 6 7` (Rand100-707, 50-100 m, both directions) and
    `--maps IMS --seeds 301`. Seed 3 (Rand3xx) and _w301 are held out (is_holdout): eval and the map tool test.

    Names: <map>_s<target>_w<seed>[_rev]... -> utils/track_gen.variant_map / train_ppo_s2r.track_group still give
    the map name. Kept out of installation/maps_gen (no generator-signature change, not in the default training
    lists); used by tools/sim_map_test.py and, when chosen, by the training scripts.
'''
import argparse
import os

import numpy as np

import utils.track_gen as tg
from utils.raceline import resample_closed, curvature_smooth

SHORT_DIR = os.path.join(os.path.dirname(tg.GEN_DIR), 'maps_short')
MIN_RADIUS = 1.6          # [m] centerline radius after scaling (the car turns at >= 0.62 m; race-line margin 0.55)


HOLDOUT_RANDOM_SEED = 3   # Rand3xx courses (and F1 maps scaled with seed 301) are kept out of training: eval only


def is_holdout(path):
    n = os.path.basename(path)
    return n.startswith(f'Rand{HOLDOUT_RANDOM_SEED}') or '_w301' in n


def short_tracks(holdout):
    '''(sorted) .npz paths in SHORT_DIR: the held-out eval courses (holdout=True) or the training ones'''
    import glob
    return sorted(t for t in glob.glob(os.path.join(SHORT_DIR, '*.npz')) if is_holdout(t) == holdout)


def scale_for(map_name, target):
    xy, L = resample_closed(tg.load_centerline(map_name), 0.5)
    k = np.max(np.abs(curvature_smooth(xy, span=2.0)))
    s = target / L
    r_min = 1.0 / max(k, 1e-6) * s
    return s, L, r_min


def make_short(map_name, seed, target=90.0, reverse=False):
    s, L, r_min = scale_for(map_name, target)
    if r_min < MIN_RADIUS:
        return None
    orig = tg.load_centerline
    tg.load_centerline = lambda m: orig(m) * s
    try:
        v = tg.make_variant(map_name, seed, reverse=reverse)
    finally:
        tg.load_centerline = orig
    v['name'] = f"{map_name}{'_rev' if reverse else ''}_s{int(target)}_w{seed}"
    return v


def random_centerline(seed, target, min_radius=1.0, tries=200):
    '''a random closed course like an F1TENTH hall track: a perturbed ellipse through 7-12 control points,
    smoothed, scaled to `target` m, tightest radius >= min_radius, no part of it closer than 4 m to another'''
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        n = rng.integers(7, 13)
        th = np.sort(rng.uniform(0, 2 * np.pi, n))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) * 0.6 + th * 0.4        # not too uneven
        ax, ay = rng.uniform(1.0, 2.2), 1.0
        r = rng.uniform(0.6, 1.4, n)
        pts = np.column_stack([ax * r * np.cos(th), ay * r * np.sin(th)])
        cs = CubicSpline(np.r_[th, 2 * np.pi], np.vstack([pts, pts[:1]]), bc_type='periodic')
        xy = cs(np.linspace(0, 2 * np.pi, 400, endpoint=False))
        xy, L = resample_closed(xy, 0.05)
        xy = xy * (target / L)
        xy, L = resample_closed(xy, 0.5)
        if np.max(np.abs(curvature_smooth(xy, span=2.0))) > 1.0 / min_radius:
            continue
        clear = tg._self_clearance(xy, np.arange(len(xy)) * (L / len(xy)), L, min_ds=8.0)
        if clear.min() < 4.0:
            continue
        return xy
    return None


def make_random(seed, target, reverse=False):
    xy = random_centerline(seed, target)
    if xy is None:
        return None
    name = f'Rand{seed}'
    orig = tg.load_centerline
    tg.load_centerline = lambda m: xy
    try:
        v = tg.make_variant(name, seed, reverse=reverse)
    finally:
        tg.load_centerline = orig
    v['name'] = f"{name}{'_rev' if reverse else ''}_s{int(target)}_w{seed}"
    v['map'] = name
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=[301])
    ap.add_argument('--target', type=float, default=90.0, help='[m] lap length')
    ap.add_argument('--out', default=SHORT_DIR)
    ap.add_argument('--random', type=int, default=0, help='also N random F1TENTH-like courses (50-100 m) per seed')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for seed in a.seeds:
        for i in range(a.random):
            rs = seed * 100 + i
            L = float(np.random.default_rng(rs).uniform(50, 100))
            for rev in (False, True):
                v = make_random(rs, L, rev)
                if v is None:
                    print(f'[skip] random course {rs}: no valid shape')
                    break
                tg.save_variant(v, a.out)
                print(f"[track] {v['name']}: {float(v['length']):.1f} m (random course)")
    if a.random and a.maps is None:
        return
    maps = a.maps or sorted(d for d in os.listdir(tg.MAP_DIR) if os.path.isfile(os.path.join(tg.MAP_DIR, d, f'{d}_centerline.csv')))
    os.makedirs(a.out, exist_ok=True)
    for m in maps:
        s, L, r_min = scale_for(m, a.target)
        if r_min < MIN_RADIUS:
            print(f'[skip] {m}: {L:.0f} m -> {a.target:.0f} m makes its tightest corner r {r_min:.2f} m < {MIN_RADIUS}')
            continue
        for seed in a.seeds:
            for rev in (False, True):
                v = make_short(m, seed, a.target, rev)
                tg.save_variant(v, a.out)
                print(f"[track] {v['name']}: {float(v['length']):.1f} m (scale {s:.3f}, tightest r {r_min:.2f} m)")


if __name__ == '__main__':
    main()

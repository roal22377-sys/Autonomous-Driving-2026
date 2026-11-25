#!/usr/bin/env python3
'''
Check the map tool against ground truth: drive a mapping run in the simulator on tracks whose walls are known,
record it like the car does (noisy 10 m LiDAR, drifting wheel odometry), run tools/map_offline.py and
tools/map_to_track.py on the recording, and measure the errors against the real track.

    python3 tools/sim_map_test.py [track.npz ...] [--out DIR] [--laps 3] [--cap 2] [--workers 2]

Driver: IL 11 at --cap m/s (as on Thursday, agent_mapping.yaml); the expert if IL 11 does not finish the laps.
Odometry like the car's VESC odom: speed x1.03, yaw rate x0.85 + 0.02 rad/s bias + noise (the 11/24 odom-only map
was a "flower" from exactly this kind of heading drift). LiDAR: 1080 beams, +-0.8 cm noise, 0.3 % dropouts, no
returns beyond 10 m (Hokuyo UST-10LX).

Per track: path error of the optimized poses (vs the true poses, after one rigid alignment), wall error (distance
of the map's wall cells to the true walls; coverage of the true walls near the path), the converted track's lap
length, centerline offset and half-width error (vs the same measurement on the true walls), and whether the
expert finishes the converted track both ways. Writes <out>/<track>/ (log, map, track) and <out>/report.md.
'''
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src/f1tenth')
sys.path[:0] = [os.path.join(REPO, 'tools'), SRC, os.path.join(SRC, 'installation')]

LASER_X_SIM = 0.17          # configs/task/dynamic.yaml lidar_pos_offset (pose = CG)
# the competition course is <= 100 m a lap (11/25): the practice course + short generated courses (utils/track_short.py,
# held-out seeds: `python -m utils.track_short --random 4 --seeds 3` and `--maps IMS --seeds 301` in src/f1tenth)
DEFAULT_TRACKS = ['installation/maps_real/Practice1124.npz', 'installation/maps_short/Rand300_s83_w300.npz',
                  'installation/maps_short/Rand302_s87_w302.npz', 'installation/maps_short/Rand303_s60_w303.npz',
                  'installation/maps_short/IMS_s90_w301.npz']


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ 1) simulated mapping run
def record(path, out, laps, cap, seed):
    os.chdir(SRC)
    import torch
    import train_il_s2r as IL
    from evaluate_s2r import model_args
    from utils.env import F1Wrapper, normalize_speed
    from utils.vectorize import RunningMeanStd
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1)
    for driver in ('imitation/11', 'expert'):
        np.random.seed(seed)
        args = IL.get_args([]) if driver == 'expert' else model_args(driver)
        args.seed, args.laps = seed, laps
        env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
        env.fixed_speed_cap = cap
        env.start_mode = 'center'
        obs, _ = env.reset(options={'track': path})
        if driver == 'expert':
            pl = RacelinePlanner(env._env.unwrapped.track, args.length_f + args.length_r, v_min=env.min_speed,
                                 v_max=env.speed_cap, max_steer=env.max_steer, max_steer_left=env.steer_left_max,
                                 max_steer_right=env.steer_right_max, **IL.expert_kwargs(env))
        else:
            rms = RunningMeanStd('agent_obs', args.obs_dim)
            rms.load(f'results/{driver}/sim2real', required=True)
            actor = torch.load(f'results/{driver}/sim2real/actor.pt', map_location='cpu')
            actor.eval()
        T, X, S, done = [], [], [], False
        while not done:
            if driver == 'expert':
                sq, vq = env.pending_commands()
                sp, s = pl.plan(env.vehicle_state(), sq, vq, env.timestep)
                a = np.clip([env.steer_command(s), normalize_speed(sp, env.min_speed, env.max_speed)], -1, 1)
            else:
                with torch.no_grad():
                    a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
            obs, r, term, trunc, info = env.step(a)
            st = env.vehicle_state()
            T.append(env.elapsed)
            X.append([st[0], st[1], st[4], st[3]])
            S.append(np.asarray(info['obs_dict']['scans'][0], np.float32))
            done = term or trunc
        if env.success:
            break
    T, X, S = np.array(T), np.array(X), np.array(S)
    rng = np.random.default_rng(seed + 1)
    # LiDAR as the Hokuyo: noise, dropouts, nothing beyond 10 m
    R = S + rng.normal(0, 0.008, S.shape).astype(np.float32)
    R[S > 10.0] = np.inf
    R[rng.random(S.shape) < 0.003] = np.nan
    fov = 4.7
    # wheel odometry with heading drift
    yaw = np.unwrap(X[:, 2])
    dt = np.diff(T, prepend=T[0] - 0.025)
    yr = np.gradient(yaw, T)
    yr_o = 0.85 * yr + 0.02 + rng.normal(0, 0.02, len(T))
    yaw_o = X[0, 2] + np.cumsum(yr_o * dt) - yr_o[0] * dt[0]
    v_o = 1.03 * X[:, 3]
    x_o = X[0, 0] + np.cumsum(v_o * np.cos(yaw_o) * dt) - v_o[0] * np.cos(yaw_o[0]) * dt[0]
    y_o = X[0, 1] + np.cumsum(v_o * np.sin(yaw_o) * dt) - v_o[0] * np.sin(yaw_o[0]) * dt[0]
    log = os.path.join(out, 'log')
    os.makedirs(log, exist_ok=True)
    np.savez_compressed(f'{log}/scan.npz', t=T, stamp=T, ranges=R, angle_min=np.float32(-fov / 2),
                        angle_increment=np.float32(fov / (S.shape[1] - 1)))
    with open(f'{log}/driving.csv', 'w') as f:
        f.write('t,mode,x,y,yaw\n')
        for t, x, y, w in zip(T, x_o, y_o, wrap(yaw_o)):
            f.write(f'{t:.4f},auto,{x:.4f},{y:.4f},{w:.5f}\n')
    np.savetxt(f'{log}/truth.csv', np.column_stack([T, X[:, :3]]), delimiter=',', header='t,x,y,yaw', comments='')
    rec = dict(driver=driver, success=bool(env.success), t=float(T[-1]),
               path=float(np.sum(np.hypot(*np.diff(X[:, :2], axis=0).T))))
    json.dump(rec, open(f'{log}/rec.json', 'w'))
    return rec


# ------------------------------------------------------------------ 3) errors against the truth
def boundary(occ_free):
    from scipy import ndimage
    wall = ~occ_free
    return wall & ndimage.binary_dilation(occ_free)


def cells_xy(mask, res, org):
    i, j = np.nonzero(mask)
    return np.column_stack([org[0] + (j + 0.5) * res, org[1] + (i + 0.5) * res])


def evaluate(path, out, rec):
    from scipy.spatial import cKDTree
    import map_offline as MO
    import map_to_track as MT
    tr = np.load(path)
    t_free = tr['occupancy'] > 128
    t_res, t_org = float(tr['resolution']), np.asarray(tr['origin'], float)[:2]
    truth = np.loadtxt(f'{out}/log/truth.csv', delimiter=',', skiprows=1)
    est = np.loadtxt(f'{out}/map/poses.csv', delimiter=',', skiprows=1)
    tt = np.column_stack([np.interp(est[:, 0], truth[:, 0], truth[:, k]) for k in (1, 2)])
    tyaw = np.interp(est[:, 0], truth[:, 0], np.unwrap(truth[:, 3]))
    # one rigid transform map frame -> true frame (least squares over all keyframes)
    A, B = est[:, 1:3], tt
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[1] *= -1
        Rm = (U @ Vt).T
    tm = mb - Rm @ ma
    to_true = lambda p: p @ Rm.T + tm
    dth = np.arctan2(Rm[1, 0], Rm[0, 0])
    pos_err = np.hypot(*(to_true(A) - B).T)
    yaw_err = np.abs(wrap(est[:, 3] + dth - tyaw))
    # walls
    g = MT.read_map(f'{out}/map')
    free_m, m_res, m_org = g
    pgm = open(f'{out}/map/map.pgm', 'rb').read()
    hdr = pgm.split(b'\n', 3)
    W, H = map(int, hdr[1].split())
    raw = np.frombuffer(hdr[3][:W * H], np.uint8).reshape(H, W)[::-1]
    mwall = cells_xy(raw == 0, m_res, m_org)
    mwall = to_true(mwall)
    tb = cells_xy(boundary(t_free), t_res, t_org)
    near_path = lambda pts, d: cKDTree(truth[:, 1:3]).query(pts, distance_upper_bound=d)[0] < d
    mw = mwall[near_path(mwall, 3.0)]
    d_wall, _ = cKDTree(tb).query(mw)
    tbn = tb[near_path(tb, 2.5)]
    cover, _ = cKDTree(mwall).query(tbn)
    # converted track vs the same measurement on the true walls
    occ, c, wl, wr = MT.build(free_m, m_res, m_org, est[:, 1:3], 2.5, 0.3)
    rl = MT.race_line(c, wl, wr)
    os.makedirs(f'{out}/track', exist_ok=True)
    name = 'Sim' + ''.join(ch for ch in os.path.basename(path)[:-4] if ch.isalnum())
    L = MT.save(f'{out}/track', name, name, False, occ, m_res, m_org, c, wl, wr, rl)
    c_true = to_true(c)
    tc = np.asarray(tr['centerline'], float)
    # true half widths measured like map_to_track on the true walls, at the true centerline
    n_t = MT.normals(tc)
    twl = MT.cast(t_free, t_res, t_org, tc, n_t, 2.5)
    twr = MT.cast(t_free, t_res, t_org, tc, -n_t, 2.5)
    from scipy import ndimage
    k = int(1.5 / 0.1) | 1
    twl = np.minimum(twl, ndimage.median_filter(twl, size=k, mode='wrap') + 0.2)
    twr = np.minimum(twr, ndimage.median_filter(twr, size=k, mode='wrap') + 0.2)
    off, j = cKDTree(tc).query(c_true)
    # (a mean of the steps of a closed loop is 0: compare the steps one by one, 11/25 review)
    same_dir = np.sum((np.roll(c_true, -1, 0) - c_true) * (np.roll(tc, -1, 0)[j] - tc[j])) >= 0
    wl_t, wr_t = (twl[j], twr[j]) if same_dir else (twr[j], twl[j])
    L_true = float(np.sum(np.hypot(*np.diff(np.vstack([tc, tc[:1]]), axis=0).T)))
    return dict(rec=rec, n_kf=len(est), pos_med=float(np.median(pos_err)), pos_max=float(pos_err.max()),
                yaw_med=float(np.degrees(np.median(yaw_err))), wall_med=float(np.median(d_wall)),
                wall_p90=float(np.percentile(d_wall, 90)), wall_p99=float(np.percentile(d_wall, 99)),
                cover=float(np.mean(cover < 0.10)), L=L, L_true=L_true, off_med=float(np.median(off)),
                off_p95=float(np.percentile(off, 95)),
                w_err_med=float(np.median(np.abs(np.r_[wl - wl_t, wr - wr_t]))),
                w_err_p95=float(np.percentile(np.abs(np.r_[wl - wl_t, wr - wr_t]), 95)),
                track=f'{out}/track/{name}.npz', plot=(mwall, tb, truth[:, 1:3], to_true(A), c_true, tc))


def plot(res, name, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mwall, tb, tpath, epath, c_true, tc = res['plot']
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(*tb.T, s=0.3, c='0.6', label='true walls')
    ax.scatter(*mwall.T, s=0.3, c='r', label='map walls (map_offline)')
    ax.plot(*tpath.T, 'k-', lw=0.4, label='true path')
    ax.plot(*epath.T, 'b-', lw=0.4, label='estimated path')
    ax.plot(*tc.T, 'g--', lw=0.8, label='true centerline')
    ax.plot(*c_true.T, 'm-', lw=0.8, label='converted centerline')
    ax.set_aspect('equal'), ax.legend(loc='upper right', markerscale=10)
    ax.set_title(f'{name}: wall error median {res["wall_med"]*100:.1f} cm, p90 {res["wall_p90"]*100:.1f} cm; '
                 f'path error median {res["pos_med"]*100:.1f} cm')
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def job(args):
    path, out, laps, cap, seed, reuse = args
    t0 = time.time()
    if reuse and os.path.exists(f'{out}/log/rec.json'):                 # map again from the same recording
        rec = json.load(open(f'{out}/log/rec.json'))
    else:
        rec = record(path, out, laps, cap, seed)
    import subprocess
    subprocess.run([sys.executable, os.path.join(REPO, 'tools/map_offline.py'), f'{out}/log', f'{out}/map',
                    '--laser_x', str(LASER_X_SIM)], check=True, stdout=subprocess.DEVNULL)
    try:
        res = evaluate(path, out, rec)
    except (Exception, SystemExit) as e:                     # report the failure, keep the other tracks
        rep = open(f'{out}/map/report.txt').read() if os.path.exists(f'{out}/map/report.txt') else ''
        return path, dict(rec=rec, error=f'{type(e).__name__}: {e}', map_report=rep, sec=time.time() - t0)
    import check_track as CT
    drv = [CT.drive('expert', res['track'], 8.0)]
    res['expert_ok'] = all(d['ok'] for d in drv)
    res['expert_t'] = drv[0]['t']
    plot(res, os.path.basename(path)[:-4], f'{out}/compare.png')
    res.pop('plot')
    res['sec'] = time.time() - t0
    return path, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tracks', nargs='*', help='track .npz (default: practice course + 4 generated, see DEFAULT_TRACKS)')
    ap.add_argument('--out', default=os.path.join(REPO, 'maps/sim_test'))
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--cap', type=float, default=2.0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--reuse_log', action='store_true', help='skip the drive when <out>/<track>/log exists (map tool changes)')
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)          # record() changes into src/f1tenth
    tracks = [os.path.abspath(t if os.path.isabs(t) or os.path.exists(t) else os.path.join(SRC, t))
              for t in (a.tracks or DEFAULT_TRACKS)]
    missing = [t for t in tracks if not os.path.exists(t)]
    if missing:
        raise SystemExit(f'missing tracks {missing}: make the short courses first (see DEFAULT_TRACKS)')
    jobs = [(t, os.path.join(a.out, os.path.basename(t)[:-4]), a.laps, a.cap, a.seed, a.reuse_log) for t in tracks]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(job, jobs)
    lines = ['# map tool against ground truth (tools/sim_map_test.py)', '',
             f'simulated mapping run: {a.laps} laps at cap {a.cap} m/s, noisy 10 m LiDAR, drifting odometry', '',
             '| track | driver | keyframes | path err median / max | heading err | wall err median / p90 / p99 | true walls covered (10 cm) | lap length map / true | centerline offset median / p95 | half-width err median / p95 | expert on the converted track |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for path, r in results:
        rec = r['rec']
        if 'error' in r:
            loops = [l for l in r['map_report'].splitlines() if l.startswith('loop closures')]
            lines.append(f'| {os.path.basename(path)[:-4]} | {rec["driver"]} | FAILED: {r["error"][:80]} | {loops[0] if loops else ""} |' + ' |' * 7)
            continue
        lines.append(f'| {os.path.basename(path)[:-4]} | {rec["driver"]}{"" if rec["success"] else " (did not finish)"} | {r["n_kf"]} | '
                     f'{r["pos_med"]*100:.1f} / {r["pos_max"]*100:.1f} cm | {r["yaw_med"]:.2f} deg | '
                     f'{r["wall_med"]*100:.1f} / {r["wall_p90"]*100:.1f} / {r["wall_p99"]*100:.1f} cm | {r["cover"]*100:.0f} % | '
                     f'{r["L"]:.1f} / {r["L_true"]:.1f} m | {r["off_med"]*100:.1f} / {r["off_p95"]*100:.1f} cm | '
                     f'{r["w_err_med"]*100:.1f} / {r["w_err_p95"]*100:.1f} cm | {"finishes" if r["expert_ok"] else "FAILS"} ({r["expert_t"]:.1f} s) |')
    text = '\n'.join(lines) + '\n'
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, 'report.md'), 'w').write(text)
    print(text)
    print(f'[saved] {a.out}/report.md and <track>/compare.png')


if __name__ == '__main__':
    main()

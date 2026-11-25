#!/usr/bin/env python3
'''
Drive a converted real-course track (tools/map_to_track.py) in the simulator: the expert, and optionally models,
2 laps from the centerline start in the current car conditions (configs/task/f1tenth.yaml sim2real, nominal).

    python3 tools/check_track.py src/f1tenth/installation/maps_real/Practice1127.npz [--models ppo_s2r/12190 ...]
        [--caps 8 4 2] [--car_lap 12.3]

Prints success / 2-lap time per driver and cap and writes <track>_drive.png (paths on the map).
--car_lap: the car's measured lap time [s] at the same cap, to compare (e.g. from tools/eval_drive.py).
A converted track is ready when the expert finishes both directions at every cap without touching a wall.
'''
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src/f1tenth')
sys.path[:0] = [SRC, os.path.join(SRC, 'installation')]
import numpy as np


def drive(model, path, cap, seed=0):
    os.chdir(SRC)                     # configs/ and results/ are relative to src/f1tenth (only inside the workers:
    import torch                      # a chdir at import time broke the spawned workers' re-import of this script)
    import train_il_s2r as IL
    from evaluate_s2r import model_args
    from utils.env import F1Wrapper, normalize_speed
    from utils.vectorize import RunningMeanStd
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1)
    np.random.seed(seed)
    args = IL.get_args([]) if model == 'expert' else model_args(model)
    args.seed, args.laps = seed, 2
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
    env.fixed_speed_cap = cap
    env.start_mode = 'center'
    obs, _ = env.reset(options={'track': path})
    if model == 'expert':
        pl = RacelinePlanner(env._env.unwrapped.track, args.length_f + args.length_r, v_min=env.min_speed,
                             v_max=env.speed_cap, max_steer=env.max_steer, max_steer_left=env.steer_left_max,
                             max_steer_right=env.steer_right_max, **IL.expert_kwargs(env))
    else:
        rms = RunningMeanStd('agent_obs', args.obs_dim)
        rms.load(f'results/{model}/sim2real', required=True)
        actor = torch.load(f'results/{model}/sim2real/actor.pt', map_location='cpu')
        actor.eval()
    done, xy, v = False, [], []
    while not done:
        if model == 'expert':
            sq, vq = env.pending_commands()
            sp, s = pl.plan(env.vehicle_state(), sq, vq, env.timestep)
            a = np.clip([env.steer_command(s), normalize_speed(sp, env.min_speed, env.max_speed)], -1, 1)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
        obs, r, term, trunc, info = env.step(a)
        st = env.vehicle_state()
        xy.append(st[:2].copy())
        v.append(float(st[3]))
        done = term or trunc
    laps = list(env.lap_times) if getattr(env, 'lap_times', None) else []
    return dict(model=model, track=os.path.basename(path)[:-4], cap=cap, seed=seed, ok=bool(env.success),
                crash=bool(env.collision), t=float(env.elapsed), laps=laps, xy=np.array(xy), v=float(np.mean(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track', help='<name>.npz (its _rev.npz is driven too when it exists)')
    ap.add_argument('--models', nargs='*', default=[], help='results/<model> names, e.g. ppo_s2r/12190')
    ap.add_argument('--caps', type=float, nargs='*', default=[8.0, 4.0, 2.0])
    ap.add_argument('--car_lap', type=float, default=None, help='[s] the car\'s lap time, for comparison')
    ap.add_argument('--workers', type=int, default=4, help='parallel simulator runs (keep low while training runs)')
    ap.add_argument('--seeds', type=int, default=1, help='runs per driver / direction / cap (start noise, LiDAR noise)')
    a = ap.parse_args()
    track = os.path.abspath(a.track)
    tracks = [track] + ([track[:-4] + '_rev.npz'] if os.path.exists(track[:-4] + '_rev.npz') else [])
    jobs = [(m, t, c, k) for t in tracks for m in ['expert'] + a.models for c in a.caps for k in range(a.seeds)]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(min(len(jobs), a.workers)) as pool:
        res = pool.starmap(drive, jobs)
    print(f'{"driver":16s} {"track":18s} {"cap":>4s}  finished  2-lap time (finished)  mean speed  failures at [s]')
    keys = sorted({(r['model'], r['track'], r['cap']) for r in res}, key=lambda k: (['expert'] + a.models).index(k[0]))
    for m, t, c in keys:
        R = [r for r in res if (r['model'], r['track'], r['cap']) == (m, t, c)]
        ok = [r for r in R if r['ok']]
        tt = f'{np.mean([r["t"] for r in ok]):6.2f} s' if ok else '     -  '
        fails = ', '.join(f'{r["t"]:.1f}{"c" if r["crash"] else ""}' for r in R if not r['ok'])
        print(f'{m:16s} {t:18s} {c:4.1f}  {len(ok)}/{len(R):<6d}  {tt:21s}  {np.mean([r["v"] for r in R]):.2f} m/s    {fails}')
    if a.car_lap:
        print(f'car lap time (given): {a.car_lap:.2f} s')
    exp_ok = all(r['ok'] for r in res if r['model'] == 'expert')
    print('[ok] the expert finishes everywhere: the track is usable' if exp_ok else
          '[check] the expert did not finish everywhere: look at the _check.png / _drive.png (narrow spot, leak, wrong wall)')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(track)
    occ, res_, org = z['occupancy'], float(z['resolution']), z['origin']
    H, W = occ.shape
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(occ, origin='lower', cmap='gray', extent=[org[0], org[0] + W * res_, org[1], org[1] + H * res_])
    ax.plot(*z['raceline'].T, 'y--', lw=0.8, label='race line')
    for r in res:
        if r['track'] == os.path.basename(track)[:-4] and r['cap'] == max(a.caps) and r['seed'] == 0:
            ax.plot(*r['xy'].T, lw=0.9, label=f'{r["model"]} ({"ok" if r["ok"] else "crash"}, {r["t"]:.1f} s)')
    ax.set_aspect('equal'), ax.legend(loc='upper right'), ax.set_title(f'{os.path.basename(track)} cap {max(a.caps):g}')
    out = track[:-4] + '_drive.png'
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'[saved] {out}')


if __name__ == '__main__':
    main()

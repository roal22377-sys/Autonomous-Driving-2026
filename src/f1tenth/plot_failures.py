'''
    Where and how a model fails: drives evaluation runs (2 laps, real-car nominal conditions) and draws
    every failed run on the track map. Run from src/f1tenth:

        python plot_failures.py imitation/9                          # eval tracks, the IL eval settings
        python plot_failures.py imitation/9 --tracks open:big        # open-space test tracks (eval_open_space.py)
        python plot_failures.py imitation/9 --tracks 'installation/maps_gen/Sochi*.npz' --caps 5
        python plot_failures.py ppo_s2r/1 --all                      # also draw the runs that finished

    Output: results/<model>/failures/<track>@<cap>_<start>.png and an overview failures/summary.png
    Each picture: the whole track with the path colored by speed and the crash marked, and a zoom on the
    last 4 s before the crash (path colored by steering command, the race line, run-off bays outlined),
    plus speed / steering over time with the crash moment. The table printed at the end says how fast the
    car was, how hard it steered and whether it was inside a run-off or a side pocket when it crashed.
'''
import argparse
import glob
import os
import sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np


def run(job):
    model, path, cap, start = job
    import torch
    import train_il_s2r as IL
    from evaluate_s2r import model_args
    from utils.env import F1Wrapper
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    seed = sum(map(ord, os.path.basename(path)))
    np.random.seed(seed)
    args = model_args(model)
    args.seed = seed
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
    env.fixed_speed_cap = cap
    env.start_mode = start
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(f'results/{model}/sim2real', required=True)
    actor = torch.load(f'results/{model}/sim2real/actor.pt', map_location='cpu')
    actor.eval()
    obs, _ = env.reset(options={'track': path})
    xs, ys, vs, steer, ey = [], [], [], [], []
    done = False
    while not done:
        with torch.no_grad():
            a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
        st = env.vehicle_state()
        obs, r, term, trunc, info = env.step(a)
        xs.append(st[0]); ys.append(st[1]); vs.append(st[3]); steer.append(float(np.clip(a[0], -1, 1)) * env.max_steer)
        ey.append(float(env.position_frenet[1]))
        done = term or trunc
    L = float(env._env.unwrapped.track.centerline.spline.s[-1])
    return dict(model=model, path=path, track=os.path.basename(path)[:-4], cap=cap, start=start, ok=bool(env.success),
                crash=bool(env.collision), progress=float(env.progress) / L, t=float(env.elapsed), dt=env.timestep,
                x=np.array(xs), y=np.array(ys), v=np.array(vs), steer=np.array(steer), ey=np.array(ey),
                max_steer=float(env.max_steer))


def where(z, x, y, ey_last, s_idx):
    '''crash location class like eval_open_space.py: run-off / pocket / corridor'''
    wl, wr = z['w_left'], z['w_right']
    outside = ey_last > wl[s_idx] + 0.05 or ey_last < -(wr[s_idx] + 0.05)
    for x0, y0, yaw, L, lo, hi in z['runoffs'] if 'runoffs' in z.files else []:
        dx, dy = x - x0, y - y0
        u, v = dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
        if outside and 0 <= u <= L and lo - 0.05 <= v <= hi + 0.05:
            return 'run-off'
    return 'pocket' if outside else 'corridor'


def draw(r, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    z = np.load(r['path'])
    occ, res, ox, oy = z['occupancy'], float(z['resolution']), float(z['origin'][0]), float(z['origin'][1])
    ext = [ox, ox + occ.shape[1] * res, oy, oy + occ.shape[0] * res]
    rl = z['raceline']
    n = len(r['x'])
    k0 = max(0, n - int(4.0 / r['dt']))
    # crash place
    cen = z['centerline']
    i_c = int(np.argmin(np.sum((cen - [r['x'][-1], r['y'][-1]]) ** 2, axis=1)))
    kind = where(z, r['x'][-1], r['y'][-1], r['ey'][-1], i_c) if r['crash'] else ''
    r['where'] = kind
    fig = plt.figure(figsize=(14, 9))
    a = fig.add_subplot(2, 2, 1)
    a.imshow(occ, cmap='gray', origin='lower', extent=ext, vmin=0, vmax=255)
    sc = a.scatter(r['x'], r['y'], c=r['v'], s=1, cmap='viridis', vmin=0, vmax=8)
    a.plot(r['x'][0], r['y'][0], 'go', ms=6, label='start')
    if r['crash']:
        a.plot(r['x'][-1], r['y'][-1], 'rx', ms=12, mew=3, label=f'crash ({kind})')
    fig.colorbar(sc, ax=a, label='speed [m/s]')
    a.set(title=f"{r['track']} cap {r['cap']:g} {r['start']}: {'finished' if r['ok'] else 'FAILED'} "
                f"at {r['progress']*100/2:.0f} % of 2 laps, {r['t']:.1f} s", aspect='equal')
    a.legend(fontsize=8)
    # zoom on the last 4 s
    a = fig.add_subplot(2, 2, 2)
    cx, cy = r['x'][-1], r['y'][-1]
    a.imshow(occ, cmap='gray', origin='lower', extent=ext, vmin=0, vmax=255)
    a.plot(rl[:, 0], rl[:, 1], 'c-', lw=1, label='race line (expert)')
    for x0, y0, yaw, L, lo, hi in z['runoffs'] if 'runoffs' in z.files else []:
        c, s = np.cos(yaw), np.sin(yaw)
        corners = [(x0 + u * c - v * s, y0 + u * s + v * c) for u, v in ((0, lo), (L, lo), (L, hi), (0, hi))]
        a.add_patch(Polygon(corners, fill=False, ec='orange', lw=1.2, ls='--'))
    sc = a.scatter(r['x'][k0:], r['y'][k0:], c=r['steer'][k0:], s=6, cmap='coolwarm', vmin=-r['max_steer'], vmax=r['max_steer'])
    if r['crash']:
        a.plot(cx, cy, 'rx', ms=14, mew=3)
    fig.colorbar(sc, ax=a, label='steering command [rad] (+ = left)')
    a.set(xlim=(cx - 6, cx + 6), ylim=(cy - 6, cy + 6), aspect='equal', title='last 4 s (orange = run-off bay)')
    a.legend(fontsize=8, loc='upper right')
    # time series
    t = np.arange(n) * r['dt']
    a = fig.add_subplot(2, 1, 2)
    a.plot(t, r['v'], label='speed [m/s]')
    a.plot(t, r['steer'] * 10, label='steering x10 [rad]')
    a.plot(t, r['ey'], label='offset from centerline [m]')
    if r['crash']:
        a.axvline(t[-1], color='r', ls='--', label='crash')
    a.set(xlim=(max(0, t[-1] - 20), t[-1] + 0.5), xlabel='time [s]', title='last 20 s')
    a.grid(alpha=0.3)
    a.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{r['track']}@{r['cap']:g}_{r['start']}.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--tracks', default='eval', help="'eval' (IL eval tracks), 'open:<level>' or a glob of .npz files")
    p.add_argument('--caps', nargs='+', type=float, default=None, help='speed caps (default: the IL eval settings)')
    p.add_argument('--start', default=None, choices=['center', 'random'], help='with --caps (default center)')
    p.add_argument('--all', action='store_true', help='also draw finished runs')
    p.add_argument('--max_plots', type=int, default=40)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    import train_il_s2r as IL
    if a.tracks == 'eval':
        allt = IL.list_variants(IL.MAPS)
        tracks = sorted([t for t in allt if IL.variant_map(t) in IL.EVAL_MAPS and
                         (IL.HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in IL.EVAL_MAPS for s in ('', '_rev')])])[:44]
    elif a.tracks.startswith('open:'):
        from eval_open_space import TEST_DIR
        tracks = sorted(glob.glob(os.path.join(TEST_DIR, a.tracks[5:], '*.npz')))
    else:
        tracks = sorted(glob.glob(a.tracks))
    if not tracks:
        sys.exit(f'no tracks for {a.tracks}')
    settings = [(c, a.start or 'center') for c in a.caps] if a.caps else \
        ([(8.0, 'center'), (5.0, 'center')] if a.tracks.startswith('open:') else list(IL.EVAL_SETTINGS))
    jobs = [(a.model, t, c, s) for c, s in settings for t in tracks]
    print(f'{a.model}: {len(jobs)} runs ({len(tracks)} tracks x {settings}), 2 laps, eval conditions', flush=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    out_dir = f'results/{a.model}/failures'
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in res if a.all or not r['ok']][:a.max_plots]
    for r in todo:
        draw(r, out_dir)
    # overview
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fails = [r for r in res if not r['ok']]
    for r in fails:                      # crash location also for the runs that were not drawn
        if 'where' not in r and r['crash']:
            z = np.load(r['path'])
            i_c = int(np.argmin(np.sum((z['centerline'] - [r['x'][-1], r['y'][-1]]) ** 2, axis=1)))
            r['where'] = where(z, r['x'][-1], r['y'][-1], r['ey'][-1], i_c)
    print(f'\nsuccess {sum(r["ok"] for r in res)}/{len(res)} | failures drawn: {sum(1 for r in todo if not r["ok"])} -> {out_dir}/')
    print(f'{"track":26s} {"cap/start":12s} {"at % of 2 laps":>14s} {"time":>6s} {"speed":>6s} {"|steer|":>8s} {"where":>9s}')
    for r in fails:
        k = max(1, len(r['v']) - 2)
        print(f"{r['track']:26s} {r['cap']:g}/{r['start']:9s} {r['progress']*50:13.0f}% {r['t']:5.1f}s "
              f"{r['v'][k]:5.2f}  {abs(r['steer'][-1]):7.3f}  {r.get('where', ''):>9s}")
    if fails:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist([r['progress'] * 50 for r in fails], bins=20)
        ax[0].set(title='where in the 2 laps it failed', xlabel='% of 2 laps')
        ax[1].hist([r['v'][max(1, len(r['v']) - 2)] for r in fails], bins=15)
        ax[1].set(title='speed just before the crash', xlabel='m/s')
        kinds = [r.get('where', '') or 'not drawn' for r in fails]
        names = sorted(set(kinds))
        ax[2].bar(names, [kinds.count(k) for k in names])
        ax[2].set(title='crash location')
        fig.tight_layout()
        fig.savefig(f'{out_dir}/summary.png', dpi=90)
        print(f'[saved] {out_dir}/summary.png')


if __name__ == '__main__':
    main()

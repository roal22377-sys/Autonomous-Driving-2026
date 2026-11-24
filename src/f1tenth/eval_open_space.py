'''
    Open-space check: does a policy still turn at a corner when a wide dead-end area continues
    straight ahead? (11/23 real car: in such places it drove into the open area and hit its end wall.)
    Run from src/f1tenth:

        python -u eval_open_space.py --make                                   # build the test tracks once
        python -u eval_open_space.py --make_train                             # training tracks of the same kind (PPO 5)
        python -u eval_open_space.py expert imitation/8 imitation/83 [--caps 5 8] [--levels train big]

    Test tracks (installation/maps_open_test/<level>/, never used for training): the evaluation maps,
    both directions, new random seeds, and a dead-end bay ("run-off") at EVERY corner tighter than the
    level's radius, straight ahead of the corner entry on the outside of the turn:
        none  : control, no run-offs. The corridor (widths, pockets, walls) is identical in all levels
                (run-offs are drawn last), so crash(big) - crash(none) is what the open areas cause:
                a car that simply fails to turn would also end up "in the run-off" straight ahead
        train : run-off sizes of the training generator (2-7 m long, 0-2.5 m wider than the track)
        big   : only the large ones (4-7 m long, 1-2.5 m wider), also at gentler corners (radius up to 8 m)
                -> what made IL run 8 (trained with 2-5 m / 0-1 m bays) drive in; PPO model selection uses it
    Each run is two laps (the success criterion everywhere) from the centerline at the nominal real-car conditions (sim_mode 'eval': delay,
    grip, noise of configs/task/f1tenth.yaml) with the steering physically limited to the measured
    full lock (0.41 rad both sides) for every model. Per crash we record where it happened:
        runoff   - inside a run-off, outside the normal corridor   (the 11/23 failure)
        pocket   - outside the corridor in a side pocket
        corridor - inside the normal corridor (an ordinary wall hit)
    and we count run-off entries (the car left the corridor into a run-off, crash or not).
    `expert` = the rule-based race-line planner that IL imitates (shows the tracks are drivable).
'''
import argparse
import os
import sys
import glob
import json
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np

TEST_DIR = os.path.join(HERE, 'installation', 'maps_open_test')
EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']
SEEDS = (901, 902)                     # training uses 100-115 (maps_gen) and 200-203 (maps_open_train)
# training tracks of the same kind (train_ppo_s2r.py --open_train_share): ALL maps, other seeds, so the test
# tracks above stay unseen; seeds 200-201 at the 'big' level, 202-203 at the 'train' level with a bay at every corner
TRAIN_DIR = os.path.join(HERE, 'installation', 'maps_open_train')
TRAIN_SEEDS = {200: 'big', 201: 'big', 202: 'train', 203: 'train'}
LEVELS = {
    'none':  dict(RUNOFF_PROB=0.0),         # control: the same corridors without any run-off
    'train': dict(RUNOFF_PROB=1.0, RUNOFF_MIN_CURV=1 / 5.0, RUNOFF_LEN=(2.0, 7.0), RUNOFF_EXTRA_W=(0.0, 2.5)),
    'big':   dict(RUNOFF_PROB=1.0, RUNOFF_MIN_CURV=1 / 8.0, RUNOFF_LEN=(4.0, 7.0), RUNOFF_EXTRA_W=(1.0, 2.5)),
}
FULL_LOCK = 0.41                       # [rad] measured 11/23, both sides (configs/task/f1tenth.yaml max_steer)


# ------------------------------------------------------------------ test tracks
def _make_job(job):
    level, map_name, seed, reverse = job
    from utils import track_gen as tg
    for k, v in LEVELS[level].items():          # this process only (fork pool)
        setattr(tg, k, v)
    v = tg.make_variant(map_name, seed, reverse, original=False)
    path = tg.save_variant(v, out_dir=os.path.join(TEST_DIR, level))
    return path, len(v['runoffs'])


def _make_train_job(job):
    level, map_name, seed, reverse = job
    from utils import track_gen as tg
    for k, v in LEVELS[level].items():
        setattr(tg, k, v)
    out = os.path.join(TRAIN_DIR, f"{map_name}{'_rev' if reverse else ''}_w{seed}.npz")
    if os.path.isfile(out) and not tg.stale_variants([out]):
        return out, len(np.load(out)['runoffs']), 'kept'
    v = tg.make_variant(map_name, seed, reverse, original=False)
    return tg.save_variant(v, out_dir=TRAIN_DIR), len(v['runoffs']), 'new'


def make_train_tracks(workers):
    from train_il_s2r import MAPS
    jobs = [(lv, m, s, r) for s, lv in TRAIN_SEEDS.items() for m in MAPS for r in (False, True)]
    with mp.get_context('fork').Pool(workers) as p:
        for i, (path, n, what) in enumerate(p.imap_unordered(_make_train_job, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {os.path.basename(path)}: {n} run-offs ({what})', flush=True)


def make_tracks(workers, levels):
    jobs = [(lv, m, s, r) for lv in levels for m in EVAL_MAPS for s in SEEDS for r in (False, True)]
    with mp.get_context('fork').Pool(workers) as p:
        for i, (path, n) in enumerate(p.imap_unordered(_make_job, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {os.path.relpath(path, TEST_DIR)}: {n} run-offs', flush=True)


# ------------------------------------------------------------------ one run
def _runoff_mask(runoffs, x, y):
    for x0, y0, yaw, L, lo, hi in runoffs:
        dx, dy = x - x0, y - y0
        u = dx * np.cos(yaw) + dy * np.sin(yaw)
        v = -dx * np.sin(yaw) + dy * np.cos(yaw)
        if 0.0 <= u <= L and lo - 0.05 <= v <= hi + 0.05:
            return True
    return False


def run(job):
    model, path, cap = job
    import torch
    import train_il_s2r as IL
    from evaluate_s2r import model_args
    from utils.env import F1Wrapper, normalize_speed
    from utils.rule_based import RacelinePlanner
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    seed = sum(map(ord, os.path.basename(path)))
    np.random.seed(seed)
    if model == 'expert':
        args = IL.get_args([])
    else:
        args = model_args(model)
    args.seed = seed
    args.laps = 2
    env = F1Wrapper(args, EVAL_MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
    env.base_params['s_min'], env.base_params['s_max'] = -FULL_LOCK, FULL_LOCK     # the real car's full lock
    env.fixed_speed_cap = cap
    env.start_mode = 'center'
    z = np.load(path)
    runoffs, wl, wr, n_c = z['runoffs'], z['w_left'], z['w_right'], len(z['w_left'])
    if model != 'expert':
        rms = RunningMeanStd('agent_obs', args.obs_dim)
        rms.load(f'results/{model}/sim2real', required=True)
        actor = torch.load(f'results/{model}/sim2real/actor.pt', map_location='cpu')
        actor.eval()
    obs, info = env.reset(options={'track': path})
    if model == 'expert':
        pl = RacelinePlanner(env._env.unwrapped.track, args.length_f + args.length_r, v_min=env.min_speed,
                             v_max=env.speed_cap, max_steer=env.max_steer, max_steer_left=env.steer_left_max,
                             **IL.expert_kwargs(env))
    L = float(env._env.unwrapped.track.centerline.spline.s[-1])
    done, entries, inside_prev, speeds = False, 0, False, []
    where = 'corridor'
    while not done:
        if model == 'expert':
            sq, vq = env.pending_commands()
            v, s = pl.plan(env.vehicle_state(), sq, vq, env.timestep)
            act = np.clip([s / env.max_steer, normalize_speed(v, env.min_speed, env.max_speed)], -1, 1)
        else:
            with torch.no_grad():
                act = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        st = env.vehicle_state()
        speeds.append(float(st[3]))
        s_f, ey = float(env.position_frenet[0]), float(env.position_frenet[1])
        i = int(round(s_f / L * n_c)) % n_c
        outside = ey > wl[i] + 0.05 or ey < -(wr[i] + 0.05)
        in_runoff = outside and _runoff_mask(runoffs, st[0], st[1])
        if in_runoff and not inside_prev:
            entries += 1
        inside_prev = in_runoff
        if env.collision:
            where = 'runoff' if in_runoff else ('pocket' if outside else 'corridor')
    crashed = bool(env.collision)
    return dict(model=model, level=os.path.basename(os.path.dirname(path)), track=os.path.basename(path)[:-4],
                cap=cap, success=bool(env.success), crash=crashed, where=where if crashed else '',
                crash_speed=speeds[-2] if crashed and len(speeds) > 1 else float('nan'), entries=entries,
                n_runoffs=int(len(runoffs)), time=float(env.elapsed), progress=float(env.progress) / L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='*', help="'expert' and/or results/<model> names, e.g. imitation/8")
    p.add_argument('--make', action='store_true', help='build the test tracks')
    p.add_argument('--make_train', action='store_true', help='build the open-space TRAINING tracks (installation/maps_open_train)')
    p.add_argument('--levels', nargs='+', default=list(LEVELS))
    p.add_argument('--caps', nargs='+', type=float, default=[5.0, 8.0], help='speed limits [m/s] (agent.yaml speed_limit)')
    p.add_argument('--workers', type=int, default=22)
    p.add_argument('--out', default=None, help='save all runs as json')
    p.add_argument('--max_tracks', type=int, default=0, help='first N tracks per level (smoke tests)')
    a = p.parse_args()
    if a.make_train:
        make_train_tracks(a.workers)
        if not a.models and not a.make:
            return
    if a.make:
        make_tracks(a.workers, a.levels)
        if not a.models:
            return
    tracks = []
    for lv in a.levels:
        tt = sorted(glob.glob(os.path.join(TEST_DIR, lv, '*.npz')))
        tracks += tt[:a.max_tracks] if a.max_tracks else tt
    if not tracks:
        sys.exit(f'no test tracks in {TEST_DIR}: run with --make first')
    jobs = [(m, t, c) for m in a.models for c in a.caps for t in tracks]
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    n_ro = {lv: sum(r['n_runoffs'] for r in res if r['level'] == lv and r['model'] == a.models[0] and r['cap'] == a.caps[0])
            for lv in a.levels}
    print(f'\n{len(tracks)} test tracks ({", ".join(f"{lv}: {n_ro[lv]} run-offs" for lv in a.levels)}), 2 laps each, '
          f'{time.time() - t0:.0f}s')
    print(f'{"model":16s} {"level":6s} {"cap":>4s} {"success":>8s} {"crash":>6s} | {"in run-off":>10s} {"pocket":>7s} '
          f'{"corridor":>9s} | {"run-off entries":>15s} {"crash speed":>11s}')
    for m in a.models:
        for lv in a.levels:
            for c in a.caps:
                rr = [r for r in res if r['model'] == m and r['level'] == lv and r['cap'] == c]
                if not rr:
                    continue
                n = len(rr)
                cnt = lambda w: sum(r['where'] == w for r in rr)
                cs = [r['crash_speed'] for r in rr if r['crash']]
                print(f'{m:16s} {lv:6s} {c:4g} {np.mean([r["success"] for r in rr])*100:7.0f}% '
                      f'{np.mean([r["crash"] for r in rr])*100:5.0f}% | {cnt("runoff"):10d} {cnt("pocket"):7d} '
                      f'{cnt("corridor"):9d} | {sum(r["entries"] for r in rr):7d} '
                      f'({sum(r["entries"] for r in rr) / max(1, n_ro[lv]) * 100:4.0f}%) '
                      f'{np.mean(cs) if cs else float("nan"):9.2f} m/s')
    print('run-off entries: drives into a run-off, as % of the run-offs on the tracks (2 laps: up to 200 %)')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        print(f'saved {a.out}')


if __name__ == '__main__':
    main()

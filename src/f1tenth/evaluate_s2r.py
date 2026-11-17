'''
    Compare models on the same tracks and simulator, 2 laps each, run from src/f1tenth:

        python -u evaluate_s2r.py imitation/6 imitation/7 ppo_s2r/1 [--modes eval ideal]

    modes
      eval  : real-car conditions at their nominal values (steering delay 0.2 s, speed delay 0.03 s,
              a_max 3 m/s^2, measured lidar/odom noise) on the evaluation tracks
              (11 eval maps x both directions x {original geometry, held-out variant w105})
      ideal : the plain simulator (no delay, a_max 9.51, no noise) on the same tracks
    Each model builds its observation from its own backup/f1tenth.yaml (old 55-dim models work).
'''
import argparse
import os
import sys
import multiprocessing as mp
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML

from utils.env import F1Wrapper
from utils.vectorize import RunningMeanStd
from utils.track_gen import list_variants, variant_map

EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']
HOLDOUT = '_w105'


def model_args(model):
    d = f'results/{model}/backup'
    args = EasyDict(YAML().load(open(f'{d}/f1tenth.yaml')))
    args.update(EasyDict(YAML().load(open('configs/task/dynamic.yaml'))))    # current car geometry
    cur = EasyDict(YAML().load(open('configs/task/f1tenth.yaml')))
    args.sim2real = cur.sim2real                                              # same real-car gaps for all
    args.laps = 2
    args.max_episode_steps = 20000
    return args


def run(job):
    model, path, mode, cap, start = job
    torch.set_num_threads(1)
    seed = sum(map(ord, os.path.basename(path)))      # same start pose / noise per track for every model
    np.random.seed(seed)
    random.seed(seed)
    args = model_args(model)
    args.seed = seed
    env = F1Wrapper(args, EVAL_MAPS, obs_noise=(mode == 'eval'), sim_mode='eval', tracks=[path])
    if mode == 'ideal':      # same track, plain simulator
        env.sim_mode = 'ideal'
    env.fixed_speed_cap = cap            # applied to the sent command like agent_node.py speed_limit
    env.start_mode = start               # 'center' or 'random' (anywhere across the track, heading error)
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(f'results/{model}/sim2real', required=True)
    actor = torch.load(f'results/{model}/sim2real/actor.pt', map_location='cpu')
    actor.eval()
    obs, info = env.reset(options={'track': path})
    done, steer, speed = False, [], []
    while not done:
        with torch.no_grad():
            a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
        obs, r, term, trunc, info = env.step(a)
        steer.append(a[0] * env.max_steer)
        speed.append(float(info['obs_dict']['linear_vels_x'][0]))
        done = term or trunc
    ok = bool(env.success)
    return (model, f'{mode}@{cap:g}/{start}', os.path.basename(path)[:-4], ok, float(env.elapsed),
            float(np.mean(speed)), float(np.abs(np.diff(steer)).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+')
    p.add_argument('--modes', nargs='+', default=['eval', 'ideal'])
    p.add_argument('--caps', nargs='+', type=float, default=[8.0, 3.0], help='speed limits [m/s] (8 = none)')
    p.add_argument('--starts', nargs='+', default=['center', 'random'])
    p.add_argument('--workers', type=int, default=22)
    p.add_argument('--max_tracks', type=int, default=0, help='limit the number of tracks (smoke tests)')
    a = p.parse_args()
    tracks = sorted([t for t in list_variants(EVAL_MAPS) if variant_map(t) in EVAL_MAPS and
                     (HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in EVAL_MAPS for s in ('', '_rev')])])
    if a.max_tracks:
        tracks = tracks[:a.max_tracks]
    jobs = [(m, t, mode, cap, st) for m in a.models for mode in a.modes for cap in a.caps for st in a.starts for t in tracks]
    a.modes = [f'{mode}@{cap:g}/{st}' for mode in a.modes for cap in a.caps for st in a.starts]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(run, jobs)
    print(f'\n{len(tracks)} tracks x 2 laps')
    print(f'{"model":16s} {"mode":17s} {"success":>9s} {"2-lap time*":>12s} {"avg speed":>10s} {"|dsteer|":>9s}')
    common = {}
    for mode in a.modes:      # lap time over tracks every model finished, so the comparison is fair
        ok_sets = [set(r[2] for r in res if r[0] == m and r[1] == mode and r[3]) for m in a.models]
        common[mode] = set.intersection(*ok_sets) if ok_sets else set()
    for m in a.models:
        for mode in a.modes:
            rr = [r for r in res if r[0] == m and r[1] == mode]
            succ = np.mean([r[3] for r in rr])
            lap = np.mean([r[4] for r in rr if r[2] in common[mode]]) if common[mode] else float('nan')
            print(f'{m:16s} {mode:17s} {succ*100:8.1f}% {lap:11.1f}s {np.mean([r[5] for r in rr]):9.2f} '
                  f'{np.mean([r[6] for r in rr]):9.4f}')
    print(f'* mean over the tracks all models finished ({ {k: len(v) for k, v in common.items()} })')
    for r in res:
        if not r[3]:
            print(f'  fail {r[0]:14s} {r[1]:17s} {r[2]:24s} at {r[4]:.1f}s')


if __name__ == '__main__':
    main()
